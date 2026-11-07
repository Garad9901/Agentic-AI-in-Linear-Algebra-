"""Hint Agent: 3 graded levels. Never states the final answer.

reject(text, claims) -> reason or None decides whether a hint may be delivered. A
rejected hint is regenerated up to 2 times, then withheld. With reject=None nothing
is checked. OURS uses certificate_rejection, whose leak check compares with the
verified Solver answer, never the stored ground truth.
"""
from __future__ import annotations

from verifier import leak_check, verify_hint

from . import prompts as P
from .base import ask_json

MAX_REGENERATIONS = 2


def _clean(reply) -> tuple[str, list]:
    if not isinstance(reply, dict):
        return "", []
    claims = reply.get("claims")
    return str(reply.get("text") or ""), claims if isinstance(claims, list) else []


def certificate_rejection(problem: dict, solution: dict):
    def reject(text: str, claims: list) -> str | None:
        if not text.strip():
            return "the hint was empty"
        leak = leak_check(text, problem["topic"], solution["answer"], claims)
        if leak.leaked:
            return f"it gives away the final answer ({leak.reason})"
        check = verify_hint(problem, text, claims)
        if not check.verified:
            return "; ".join(check.reasons)
        return None
    return reject


def _hint(problem: dict, level: int, solution, earlier: list, misconception, wrong_step, reject) -> dict:
    reference = P.compact(solution) if solution else None
    rejected = None
    for attempt in range(MAX_REGENERATIONS + 1):
        reply = ask_json("hint", P.hint(problem, level, reference, earlier, misconception, wrong_step, rejected))
        text, claims = _clean(reply)
        rejected = reject(text, claims) if reject else None
        if rejected is None:
            return {"level": level, "text": text, "claims": claims, "withheld": False, "regenerations": attempt}
    return {"level": level, "text": "", "claims": [], "withheld": True, "regenerations": MAX_REGENERATIONS,
            "reason": rejected}


def make_hints(problem: dict, solution: dict | None, misconception=None, wrong_step=None,
               reject=None, graded: bool = True) -> list:
    hints: list = []
    for level in ((1, 2, 3) if graded else (0,)):
        earlier = [h["text"] for h in hints if h["text"]]
        hints.append(_hint(problem, level, solution, earlier, misconception, wrong_step, reject))
    return hints
