"""Verification Agent: checks Solver output, with up to 2 retries, then ABSTAIN.

mode "final" checks the final-answer certificate. mode "steps" also runs the step
checker over the Solver's own working. solve_checked takes the check as a function
so B2 can use the same loop with an LLM verifier in place of the certificates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from verifier import check_steps, verify

from . import solver

MAX_RETRIES = 2
NOT_JSON = "the reply was not valid JSON"


@dataclass
class VerificationResult:
    verified: bool
    solution: dict | None
    retries: int = 0
    reasons: list = field(default_factory=list)
    unchecked_steps: int = 0
    weak_steps: int = 0


def certificate_check(problem: dict, solution, mode: str):
    """Returns (reason or None, unchecked count, weak count)."""
    if not isinstance(solution, dict):
        return NOT_JSON, 0, 0
    verdict = verify(problem["topic"], problem, solution.get("answer"))
    if not verdict.ok:
        return f"final answer: {verdict.reason}", 0, 0
    if mode != "steps":
        return None, 0, 0
    steps = solution.get("steps")
    if not isinstance(steps, list) or not steps:
        return 'the "steps" list is missing', 0, 0
    report = check_steps(problem["topic"], problem, steps)
    if report.first_bad is not None:
        return f"step {report.first_bad + 1}: {report.reason}", len(report.unchecked), len(report.weak)
    return None, len(report.unchecked), len(report.weak)


def solve_checked(problem: dict, check) -> VerificationResult:
    """check(solution) -> (reason or None, unchecked count, weak count)."""
    solution = solver.solve(problem)
    result = VerificationResult(False, None)
    for attempt in range(MAX_RETRIES + 1):
        reason, result.unchecked_steps, result.weak_steps = check(solution)
        if reason is None:
            result.verified = True
            result.solution = solution
            return result
        result.reasons.append(reason)
        if attempt == MAX_RETRIES:
            break
        result.retries += 1
        solution = solver.resolve(problem, solution, reason)
    return result  # ABSTAIN: solution stays None, so nothing reaches the student


def solve_verified(problem: dict, mode: str = "steps") -> VerificationResult:
    return solve_checked(problem, lambda solution: certificate_check(problem, solution, mode))
