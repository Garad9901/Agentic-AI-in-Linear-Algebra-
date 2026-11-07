"""Hint Agent: 3 graded levels. Never states the final answer.

review(text, claims) -> (rejection or None, extra fields) decides whether a hint may be
delivered. A rejected hint is regenerated up to 2 times, then withheld. With review=None
nothing is checked.

OURS (decision of 2026-11-07) withholds a hint only if it leaks or holds a refuted claim.
Claims that cannot be checked or are only weakly checked are delivered with an
"unverified" flag. The leak check compares with every answer the tutor has computed:
the Solver's certified answer and the Verification Agent's own SymPy computation.
It never uses the stored ground truth.
"""
from __future__ import annotations

from verifier import leak_check, verify_hint

from . import prompts as P
from .base import ask_json

MAX_REGENERATIONS = 2
_CLAIM_FLAG = {"certified": "certified", "refuted": "refuted", "weak": "unverified", "uncheckable": "unverified"}


def _clean(reply) -> tuple[str, list]:
    if not isinstance(reply, dict):
        return "", []
    claims = reply.get("claims")
    return str(reply.get("text") or ""), claims if isinstance(claims, list) else []


def certificate_review(problem: dict, answers: list):
    """answers: answers to guard against leaking. None entries are ignored."""
    answers = [a for a in answers if a is not None]

    def review(text: str, claims: list):
        if not text.strip():
            return "the hint was empty", {}
        for answer in answers:
            leak = leak_check(text, problem["topic"], answer, claims)
            if leak.leaked:
                return f"it gives away the final answer ({leak.reason})", {}
        check = verify_hint(problem, text, claims)
        claim_flags = [_CLAIM_FLAG[s] for s in check.statuses]
        if "refuted" in claim_flags:
            return "; ".join(r for r in check.reasons if "refuted" in r), {}
        return None, {"flag": "certified" if check.verified else "unverified", "claim_flags": claim_flags}
    return review


def _hint(problem: dict, level: int, reference, earlier: list, misconception, wrong_step, review) -> dict:
    reference = P.compact(reference) if reference else None
    rejected = None
    for attempt in range(MAX_REGENERATIONS + 1):
        reply = ask_json("hint", P.hint(problem, level, reference, earlier, misconception, wrong_step, rejected))
        text, claims = _clean(reply)
        rejected, extra = review(text, claims) if review else (None, {})
        if rejected is None:
            return {"level": level, "text": text, "claims": claims, "withheld": False, "regenerations": attempt,
                    **extra}
    return {"level": level, "text": "", "claims": [], "withheld": True, "regenerations": MAX_REGENERATIONS,
            "reason": rejected}


def make_hints(problem: dict, reference: dict | None, misconception=None, wrong_step=None,
               review=None, graded: bool = True) -> list:
    hints: list = []
    for level in ((1, 2, 3) if graded else (0,)):
        earlier = [h["text"] for h in hints if h["text"]]
        hints.append(_hint(problem, level, reference, earlier, misconception, wrong_step, review))
    return hints
