"""LLM wrapper: one function, call_llm(role, prompt) -> text.

Provider, model and key come from .env. Temperature is 0.7 (prereg-v1 s5). Every
response is cached on disk under a hash of (model, prompt, run), so a rerun costs
zero API calls and each of the 3 runs gets a fresh response. Rate limits and server errors back off exponentially. Every
call, cached or live, is logged to logs/calls.jsonl with role, tokens and
latency. The key and the prompt text are never logged.

Providers all speak the OpenAI /chat/completions format, so no SDK is needed.
There is deliberately no mock backend: a result row always comes from a model.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
LOG_PATH = ROOT / "logs" / "calls.jsonl"

# provider -> (base URL, fallback key env var, default requests/minute)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 25),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", 9),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", 60),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 15),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", 50),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", 25),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", 50),
}
RETRY_STATUS = (429, 500, 502, 503, 504)
DAILY_QUOTA = "PerDay"  # Gemini names per-day limits like GenerateRequestsPerDayPerProjectPerModel
CHARS_PER_TOKEN = 1.8  # measured from HTTP 413 sizes: prompts are mostly digits and brackets
TPM_MARGIN = 100

# Per thread, because run.py works on several problems at once.
_state = threading.local()
_pace_lock = threading.Lock()
_log_lock = threading.Lock()
_next_slot = 0.0


class LLMError(RuntimeError):
    pass


def _trackers() -> list:
    if not hasattr(_state, "trackers"):
        _state.trackers = []
    return _state.trackers


def current_run() -> int:
    return getattr(_state, "run", 1)


@contextmanager
def track():
    """Collect the log record of every call_llm made inside the block."""
    calls: list[dict] = []
    trackers = _trackers()
    trackers.append(calls)
    try:
        yield calls
    finally:
        trackers.remove(calls)


@contextmanager
def use_run(run: int):
    """Calls inside the block belong to this run index, which is part of the cache key."""
    previous = current_run()
    _state.run = run
    try:
        yield
    finally:
        _state.run = previous


def flag_last_call(name: str) -> None:
    """Mark the most recent tracked call, e.g. when its reply could not be parsed."""
    for calls in _trackers():
        if calls:
            calls[-1][name] = True


def _settings() -> dict:
    load_dotenv(ROOT / ".env")
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()
    if provider not in PROVIDERS:
        raise LLMError(f"LLM_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}. See .env.example.")
    if not model:
        raise LLMError("LLM_MODEL is not set. See .env.example.")
    base, key_env, rpm = PROVIDERS[provider]
    return {
        "provider": provider,
        "model": model,
        "base": base,
        "key_env": key_env,
        "rpm": float(os.environ.get("LLM_RPM") or rpm),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS") or 8192),
        "max_retries": int(os.environ.get("LLM_MAX_RETRIES") or 6),
        "max_wait": float(os.environ.get("LLM_MAX_WAIT") or 180),
        "json_mode": os.environ.get("LLM_JSON_MODE", "0") == "1",
        "tpm": int(os.environ.get("LLM_TPM") or 0),
        "temperature": float(os.environ.get("LLM_TEMPERATURE") or 0.7),
        "timeout": float(os.environ.get("LLM_TIMEOUT") or 900),
    }


def _api_key(cfg: dict) -> str:
    key = os.environ.get("LLM_API_KEY", "").strip() or os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise LLMError(f"no API key: set LLM_API_KEY in .env (or {cfg['key_env']} in the environment)")
    return key


def _reply_budget(cfg: dict, prompt: str) -> int:
    """Some free tiers refuse a request outright if prompt + max_tokens exceeds tokens per minute."""
    if not cfg["tpm"]:
        return cfg["max_tokens"]
    budget = min(cfg["max_tokens"], cfg["tpm"] - int(len(prompt) / CHARS_PER_TOKEN) - TPM_MARGIN)
    if budget < 256:
        raise LLMError(f"prompt leaves no reply budget under LLM_TPM={cfg['tpm']}")
    return budget


def cache_key(model: str, prompt: str, run: int) -> str:
    return hashlib.sha256(json.dumps([model, prompt, run], ensure_ascii=False).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_cache(key: str) -> dict | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(key: str, entry: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(key).with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_cache_path(key))


def _pace(rpm: float) -> None:
    """Hand out request slots 60/rpm apart, across all threads."""
    global _next_slot
    with _pace_lock:
        now = time.time()
        slot = max(now, _next_slot)
        _next_slot = slot + 60.0 / rpm
    if slot > now:
        time.sleep(slot - now)


def _backoff(attempt: int, retry_after: str | None, max_wait: float) -> float:
    try:
        wait = float(retry_after)
    except (TypeError, ValueError):
        wait = min(max_wait, 5.0 * 2 ** (attempt - 1))
    if wait > max_wait:
        raise LLMError(f"provider asked to wait {wait:.0f}s (over LLM_MAX_WAIT): quota is probably exhausted")
    return wait


def _request(cfg: dict, prompt: str) -> dict:
    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": cfg["temperature"],
        "max_tokens": _reply_budget(cfg, prompt),
    }
    if cfg["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {_api_key(cfg)}",
        "Content-Type": "application/json",
        "User-Agent": "vgat-benchmark/1.0",
    }
    rate_limits = timeouts = 0
    for attempt in range(1, cfg["max_retries"] + 2):
        _pace(cfg["rpm"])
        req = urllib.request.Request(f"{cfg['base']}/chat/completions", data=data, headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=cfg["timeout"]) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode(errors="replace")
            msg = body_text[:300]
            if e.code == 429 and DAILY_QUOTA in body_text:
                raise LLMError(f"daily quota exhausted for {cfg['model']}: retrying today cannot help") from None
            if e.code in RETRY_STATUS and attempt <= cfg["max_retries"]:
                rate_limits += e.code == 429
                time.sleep(_backoff(attempt, e.headers.get("Retry-After"), cfg["max_wait"]))
                continue
            raise LLMError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            timeouts += isinstance(e, TimeoutError)
            if attempt <= cfg["max_retries"]:
                time.sleep(_backoff(attempt, None, cfg["max_wait"]))
                continue
            raise LLMError(f"network error: {e}") from None
        choice = resp["choices"][0]
        usage = resp.get("usage") or {}
        return {
            "text": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_s": round(time.time() - t0, 3),
            "attempts": attempt,
            "rate_limit_errors": rate_limits,
            "timeouts": timeouts,
        }
    raise LLMError("retries exhausted")


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    for calls in _trackers():
        calls.append(record)


def call_llm(role: str, prompt: str) -> str:
    cfg = _settings()
    key = cache_key(cfg["model"], prompt, current_run())
    entry = _read_cache(key)
    cached = entry is not None
    request_s = 0.0
    if entry is None:
        t0 = time.time()
        entry = _request(cfg, prompt)
        request_s = time.time() - t0  # includes pacing and backoff, which are not compute
        # An empty reply cut off by the token limit is the model failing to answer, not an outage.
        if not entry["text"].strip() and entry["finish_reason"] != "length":
            raise LLMError(f"empty reply (finish_reason={entry['finish_reason']})")
        _write_cache(key, entry)
    # Tokens and latency are those of the original call, so T7 is the same on a cached rerun.
    _log({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "provider": cfg["provider"],
        "model": cfg["model"],
        "run": current_run(),
        "cached": cached,
        "prompt_tokens": entry.get("prompt_tokens"),
        "completion_tokens": entry.get("completion_tokens"),
        "total_tokens": entry.get("total_tokens"),  # includes thinking, which completion_tokens may not
        "latency_s": entry.get("latency_s"),
        "request_s": round(request_s, 3),
        "attempts": 0 if cached else entry.get("attempts"),
        "rate_limit_errors": 0 if cached else entry.get("rate_limit_errors"),
        "timeouts": 0 if cached else entry.get("timeouts"),
        "finish_reason": entry.get("finish_reason"),
    })
    return entry["text"]
