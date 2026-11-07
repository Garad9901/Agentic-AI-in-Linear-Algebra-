"""Gemini Batch API mode for run.py --batch.

The agents make several calls in sequence, and a batch job only answers calls it was
given, so a run works in rounds. Each round replays every problem through its system.
A reply already in the cache is used; the first missing reply stops that problem and
joins the next job. When the job finishes, its replies go into the same cache as
synchronous replies, and the next round gets further. When a round finds nothing
missing, the rows are complete and identical in form to a synchronous run.

The job name is saved before polling starts, so after a crash or sleep the same command
picks up the same job. Nothing is lost except time. A job can take up to 24 hours, and
a system whose calls depend on each other needs several rounds.

The Batch API records no latency, so these rows have none. Google's page documents two
paths for the job state and two for the results, so both are read here.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import llm

API = "https://generativelanguage.googleapis.com/v1beta"
STATE_DIR = llm.CACHE_DIR / "batches"
POLL_SECONDS = 60
MAX_REQUEST_BYTES = 15_000_000  # the documented limit for inline requests is 20 MB
MAX_KEY_FAILURES = 2
MAX_ROUNDS = 40
DONE = {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError)
_FINISH = {"STOP": "stop", "MAX_TOKENS": "length"}


class BatchError(RuntimeError):
    pass


def _http(cfg: dict, method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"x-goog-api-key": llm._api_key(cfg), "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def request_body(cfg: dict, prompt: str) -> dict:
    return {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": cfg["temperature"], "maxOutputTokens": cfg["max_tokens"]}}


def submit(cfg: dict, prompts: dict, name: str) -> str:
    """prompts is {cache key: prompt}. Returns the job name."""
    requests = [{"request": request_body(cfg, p), "metadata": {"key": k}} for k, p in prompts.items()]
    body = {"batch": {"display_name": name, "input_config": {"requests": {"requests": requests}}}}
    job = _http(cfg, "POST", f"{API}/models/{cfg['model']}:batchGenerateContent", body)
    return job["name"]


def fetch(cfg: dict, name: str) -> dict:
    return _http(cfg, "GET", f"{API}/{name}")


def job_state(job: dict) -> str:
    state = (job.get("metadata") or {}).get("state") or job.get("state") or ""
    return state.replace("JOB_STATE_", "").replace("BATCH_STATE_", "")


def inline_results(job: dict) -> list:
    for holder in (job.get("response"), job.get("dest")):
        if isinstance(holder, dict) and "inlinedResponses" in holder:
            found = holder["inlinedResponses"]
            return found.get("inlinedResponses", []) if isinstance(found, dict) else found
    return []


def to_cache_entry(result: dict) -> dict | None:
    """One inline result as a cache entry, or None if it failed."""
    response = result.get("response")
    if not isinstance(response, dict) or result.get("error"):
        return None
    candidates = response.get("candidates") or []
    if not candidates:
        return None
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    usage = response.get("usageMetadata") or {}
    reason = candidate.get("finishReason", "")
    finish = _FINISH.get(reason, reason.lower())
    if not text.strip() and finish != "length":
        return None
    return {"text": text, "finish_reason": finish, "prompt_tokens": usage.get("promptTokenCount"),
            "completion_tokens": usage.get("candidatesTokenCount"), "total_tokens": usage.get("totalTokenCount"),
            "latency_s": None, "attempts": 1, "rate_limit_errors": 0, "timeouts": 0, "batch": True}


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"job": None, "keys": [], "failures": {}}


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    llm.atomic_write(path, json.dumps(state))


def _wait(cfg: dict, name: str, poll_seconds: float) -> dict:
    last = None
    while True:
        try:
            job = fetch(cfg, name)
        except _NETWORK_ERRORS:  # for example, the machine slept: try again
            time.sleep(poll_seconds)
            continue
        state = job_state(job)
        if state != last:
            print(f"  job {name}: {state}", flush=True)
            last = state
        if state in DONE or job.get("done"):
            return job
        time.sleep(poll_seconds)


def harvest(cfg: dict, job: dict, keys: list, failures: dict) -> int:
    """Store the replies of a finished job. Keys with no reply count as failures."""
    results = inline_results(job)
    stored = 0
    answered = set()
    for position, result in enumerate(results):
        key = (result.get("metadata") or {}).get("key") or (keys[position] if position < len(keys) else None)
        entry = to_cache_entry(result)
        if key and entry:
            llm.store(cfg, key, entry)
            answered.add(key)
            stored += 1
    for key in keys:
        if key not in answered:
            failures[key] = failures.get(key, 0) + 1
    return stored


def _take(pending: dict) -> dict:
    """As many pending prompts as fit in one inline request. The rest wait for the next round."""
    taken, size = {}, 0
    for key, prompt in pending.items():
        size += len(prompt.encode("utf-8")) + 300
        if taken and size > MAX_REQUEST_BYTES:
            break
        taken[key] = prompt
    return taken


def drive(cfg: dict, label: str, replay, poll_seconds: float = POLL_SECONDS) -> list:
    """replay(collector) -> rows. Returns the rows of the first round with nothing missing."""
    path = STATE_DIR / f"{label}.json"
    state = _load(path)
    for round_no in range(1, MAX_ROUNDS + 1):
        if state["job"]:
            job = _wait(cfg, state["job"], poll_seconds)
            stored = harvest(cfg, job, state["keys"], state["failures"])
            print(f"round {round_no}: {stored} of {len(state['keys'])} replies stored", flush=True)
            state.update(job=None, keys=[])
            _save(path, state)
        collector = llm.BatchCollector({k for k, n in state["failures"].items() if n >= MAX_KEY_FAILURES})
        rows = replay(collector)
        if not collector.pending:
            path.unlink(missing_ok=True)
            return rows
        chosen = _take(collector.pending)
        state.update(job=submit(cfg, chosen, label), keys=list(chosen))
        _save(path, state)
        print(f"round {round_no}: submitted {len(chosen)} requests as {state['job']}", flush=True)
    raise BatchError(f"still incomplete after {MAX_ROUNDS} rounds")
