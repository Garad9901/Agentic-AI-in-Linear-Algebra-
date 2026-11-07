"""Verification Agent: checks Solver output, repairs, then flags or withholds each claim.

OURS (S0 in prereg-v1) asks the Solver to repair refuted claims up to 2 times. In the
final version, a claim still refuted is withheld, a certified claim is delivered, and a
claim that cannot be checked or is only weakly checked is delivered with an "unverified"
flag (decision of 2026-11-07). The final answer is one claim: if it is refuted it is
withheld, and the steps that survive are still delivered.

mode "steps" checks the working and the answer. mode "final" (ablation A2) checks the
answer only, so every step is delivered as unverified.

B2 uses solve_checked with an LLM judge in place of the certificates. It keeps its own
rule: a response the judge rejects after 2 repairs is not delivered at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from verifier import check_steps, step_statuses, verify

from . import solver

MAX_RETRIES = 2
NOT_JSON = "the reply was not valid JSON"
CERTIFIED, UNVERIFIED, WITHHELD = "certified", "unverified", "withheld"
_STEP_FLAG = {"certified": CERTIFIED, "refuted": WITHHELD, "weak": UNVERIFIED, "uncheckable": UNVERIFIED}


@dataclass
class VerificationResult:
    verified: bool
    solution: dict | None
    retries: int = 0
    reasons: list = field(default_factory=list)
    unchecked_steps: int = 0
    weak_steps: int = 0


@dataclass
class FlaggedSolution:
    solution: dict | None  # the final version, before anything is withheld
    step_flags: list = field(default_factory=list)
    answer_flag: str = WITHHELD
    retries: int = 0
    reasons: list = field(default_factory=list)

    def delivered(self) -> dict | None:
        """What the student sees: refuted steps and a refuted answer are removed."""
        if self.solution is None:
            return None
        steps = self.solution.get("steps") if isinstance(self.solution.get("steps"), list) else []
        kept = [s for s, f in zip(steps, self.step_flags) if f != WITHHELD]
        answer = self.solution.get("answer") if self.answer_flag != WITHHELD else None
        return {"steps": kept, "answer": answer}


def _answer_flag(problem: dict, answer) -> tuple:
    verdict = verify(problem["topic"], problem, answer)
    if verdict.ok:
        return CERTIFIED, None
    if verdict.reason.startswith("malformed"):
        return UNVERIFIED, None  # cannot be checked, so it is flagged, not withheld
    return WITHHELD, f"final answer: {verdict.reason}"


def assess(problem: dict, solution: dict, mode: str) -> tuple:
    """(step flags, answer flag, reasons for every refuted claim)."""
    steps = solution.get("steps") if isinstance(solution.get("steps"), list) else []
    if mode == "steps":
        step_flags = [_STEP_FLAG[s] for s in step_statuses(problem["topic"], problem, steps)]
    else:
        step_flags = [UNVERIFIED] * len(steps)
    answer_flag, answer_reason = _answer_flag(problem, solution.get("answer"))
    reasons = []
    if WITHHELD in step_flags:
        report = check_steps(problem["topic"], problem, steps)
        first = report.first_bad if report.first_bad is not None else step_flags.index(WITHHELD)
        reasons.append(f"step {first + 1}: {report.reason or 'refuted'}")
    if answer_reason:
        reasons.append(answer_reason)
    return step_flags, answer_flag, reasons


def solve_flagged(problem: dict, mode: str = "steps") -> FlaggedSolution:
    solution = solver.solve(problem)
    result = FlaggedSolution(None)
    for attempt in range(MAX_RETRIES + 1):
        if not isinstance(solution, dict):
            result.reasons.append(NOT_JSON)
            break
        result.solution = solution
        result.step_flags, result.answer_flag, reasons = assess(problem, solution, mode)
        if not reasons or attempt == MAX_RETRIES:
            result.reasons.extend(reasons)
            break
        result.reasons.extend(reasons)
        result.retries += 1
        repaired = solver.resolve(problem, solution, "; ".join(reasons))
        if not isinstance(repaired, dict):
            break  # keep the last version that could be read
        solution = repaired
    return result


def certificate_check(problem: dict, solution, mode: str):
    """Returns (reason or None, unchecked count, weak count). Used by tests and older callers."""
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
    """check(solution) -> (reason or None, unchecked count, weak count). B2 abstains on failure."""
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
    return result
