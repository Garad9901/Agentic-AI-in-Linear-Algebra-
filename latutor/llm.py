"""LLM backends.

OpenAICompatible covers every provider that exposes an OpenAI-style
/chat/completions endpoint (Groq, Google Gemini's OpenAI endpoint, OpenRouter,
Mistral, Cerebras, Together, ...). The API key is read from an environment
variable and never written to disk or logs.

MockLLM exists only to dry-run the pipeline without an API key. Every record it
produces is tagged run_mode="DRYRUN", and scripts/analyse.py refuses to build
paper tables from DRYRUN records. Never report its numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# provider name in config/models.yaml -> (base URL, API-key env var, default requests/minute)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 25),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 9),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 9),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 15),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", 50),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 25),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", 50),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN", 12),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 1
    rate_limit_errors: int = 0
    run_mode: str = "LIVE"


def extract_json(text: str):
    """Parse the first JSON object in a model reply (tolerates ```json fences and prose)."""
    if text is None:
        raise ValueError("empty reply")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise ValueError("no JSON object found in reply")


class OpenAICompatible:
    def __init__(self, provider: str, model_id: str, temperature: float = 0.7,
                 max_tokens: int = 4096, rpm: int | None = None, timeout: int = 180,
                 max_retries: int = 6, json_mode: bool = True):
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; add it to latutor/llm.py PROVIDERS")
        base, key_env, default_rpm = PROVIDERS[provider]
        self.base_url = os.environ.get(f"{provider.upper()}_BASE_URL", base)
        self.key = os.environ.get(key_env)
        if not self.key:
            raise LLMError(f"set the {key_env} environment variable (never commit keys)")
        self.provider, self.model_id = provider, model_id
        self.temperature, self.max_tokens = temperature, max_tokens
        self.min_interval = 60.0 / (rpm or int(os.environ.get(f"{provider.upper()}_RPM", default_rpm)))
        self.timeout, self.max_retries, self.json_mode = timeout, max_retries, json_mode
        self._last = 0.0

    def chat(self, system: str, user: str, seed: int | None = None) -> LLMResult:
        body = {"model": self.model_id, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if seed is not None:
            body["seed"] = seed
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode()
        rl_errors, attempt = 0, 0
        while True:
            attempt += 1
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.key}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                lat = time.time() - t0
                usage = resp.get("usage") or {}
                text = resp["choices"][0]["message"].get("content") or ""
                return LLMResult(text, lat, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                                 attempt, rl_errors)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code == 400 and self.json_mode and "response_format" in msg:
                    self.json_mode = False  # provider/model without JSON mode: fall back to prompt-only JSON
                    body.pop("response_format", None)
                    data = json.dumps(body).encode()
                    continue
                if e.code in (429, 500, 502, 503, 504) and attempt <= self.max_retries:
                    rl_errors += e.code == 429
                    retry_after = e.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit()
                               else min(120, 5 * 2 ** (attempt - 1)))
                    continue
                raise LLMError(f"HTTP {e.code}: {msg}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt <= self.max_retries:
                    time.sleep(min(120, 5 * 2 ** (attempt - 1)))
                    continue
                raise LLMError(f"network error: {e}") from None


@dataclass
class MockLLM:
    """Dry-run stand-in. Answers from the stored solution with injected mistakes. NOT A MODEL."""
    error_rate: float = 0.3
    model_id: str = "mock-dryrun"
    provider: str = "mock"
    problems: dict = field(default_factory=dict)  # id -> problem, set by the runner

    def chat(self, system: str, user: str, seed: int | None = None) -> LLMResult:
        from .grader import gold_final_claims
        rng = random.Random(hashlib.sha256((user + str(seed)).encode()).hexdigest())
        m = re.search(r"PROBLEM_ID: (\S+)", user)
        p = self.problems.get(m.group(1)) if m else None
        role = re.search(r"ROLE: (\w+)", system)
        role = role.group(1) if role else "solver"
        if role == "llm_verifier":
            n = len(re.findall(r'"type"', user.split("CLAIMS TO JUDGE:")[-1]))
            out = {"judgements": [{"index": i, "verdict": "correct" if rng.random() > 0.2 else "incorrect",
                                   "reason": "mock"} for i in range(n)]}
        elif role in ("detector",):
            out = {"first_error_step": rng.choice([None, 1, 2, 3])}
        elif role == "diagnostic":
            out = {"misconception": "MOCK_LABEL", "explanation": "mock"}
        elif role in ("hint", "hint_single"):
            k = 1 if role == "hint_single" else 3
            out = {"hints": [{"level": i + 1, "text": f"Mock hint {i + 1}: re-check the step.", "claims": []}
                             for i in range(k)]}
        elif role == "single_tutor_student":
            out = {"first_error_step": rng.choice([None, 1, 2]), "misconception": "MOCK_LABEL",
                   "hints": [{"level": i + 1, "text": "Mock hint.", "claims": []} for i in range(3)]}
        else:
            claims = gold_final_claims(p) if p else []
            claims = [_corrupt(c, rng) if rng.random() < self.error_rate else c for c in claims]
            out = {"steps": [{"explanation": "Mock step.", "claims": []}], "final_answer": claims}
        return LLMResult(json.dumps(out), 0.0, 0, 0, 1, 0, run_mode="DRYRUN")


def _corrupt(c, rng):
    c = json.loads(json.dumps(c))
    for k in ("value", "matrix", "vector", "coefficients", "values", "P"):
        if k in c:
            v = c[k]
            if isinstance(v, (int, str)):
                c[k] = str(int(str(v).split("/")[0]) + 1)
            elif isinstance(v, list) and v and isinstance(v[0], list):
                v[0][0] = str(int(str(v[0][0]).split("/")[0]) + 1)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                v[0]["value"] = str(int(str(v[0]["value"]).split("/")[0]) + 1)
            elif isinstance(v, list) and v:
                v[0] = str(int(str(v[0]).split("/")[0]) + 1)
            break
    return c


def load_models(path="config/models.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    models = cfg.get("models") or []
    problems = [i for i, m in enumerate(models)
                if not all(m.get(k) for k in ("provider", "model_id", "family", "release_date", "access_date"))]
    return models, problems
