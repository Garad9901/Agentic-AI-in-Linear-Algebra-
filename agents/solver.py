"""Solver Agent."""
from __future__ import annotations

from . import prompts as P
from .base import ask_json


def solve(problem: dict) -> dict | None:
    return ask_json("solver", P.solver(problem))


def resolve(problem: dict, previous: dict | None, reason: str) -> dict | None:
    return ask_json("solver", P.solver_retry(problem, P.compact(previous), reason))
