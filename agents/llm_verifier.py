"""LLM verifier for B2: an LLM judge in the place of the SymPy certificates.

A judge that returns no usable verdict counts as approval: it raised no objection,
and B2 has no other check.
"""
from __future__ import annotations

from . import prompts as P
from .base import ask_json
from .verification import NOT_JSON


def _reason(reply) -> str | None:
    if isinstance(reply, dict) and str(reply.get("verdict", "")).lower() == "incorrect":
        return str(reply.get("reason") or "the verifier judged it incorrect")
    return None


def check_solution(problem: dict, solution):
    """Same contract as verification.certificate_check."""
    if not isinstance(solution, dict):
        return NOT_JSON, 0, 0
    return _reason(ask_json("llm_verifier", P.llm_verify_solution(problem, solution))), 0, 0


def check_hint(problem: dict, solution: dict, text: str, claims: list) -> str | None:
    if not text.strip():
        return "the hint was empty"
    return _reason(ask_json("hint_verifier", P.llm_verify_hint(problem, solution, text, claims)))
