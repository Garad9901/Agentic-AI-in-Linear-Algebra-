"""Systems under test. Names follow prereg-v1 (see docs/DEVIATIONS.md for the changes).

  B1    single LLM: one prompt does everything
  B2    multi-agent tutor with an LLM verifier in place of SymPy
  B3    single LLM with a self-verification prompt
  B4    Diagnostic + Hint + Solver agents with no verification
  OURS  the full system (S0 in prereg-v1)
  A1-A4 ablations of OURS: A1 no Verification Agent, A2 final answer only,
        A3 no Diagnostic Agent, A4 single-level hint
  DET   zero-shot LLM first-error detector, D3 only (B4 in prereg-v1, RQ2 primary test)

run_system() returns what a system produced and what reached the student. It
never grades: grading against the stored ground truth happens in score.py.

OURS repairs, then withholds only claims still refuted and delivers the rest, with an
"unverified" flag on claims it cannot certify (decision of 2026-11-07). Its result
keeps the whole final version, with step_flags and answer_flag saying what was withheld.

D1 and D2 have no student work, so the Diagnostic Agent cannot run on them and hints
are written for a student who is stuck and has not started. Hints on D1 are exploratory
and made in run 1 only (decision of 2026-11-07). A1 is the same configuration as B4,
and A3 the same as OURS on D1 and D2; each pair is reported as one result.
"""
from __future__ import annotations

from dataclasses import dataclass

import llm
from agents import diagnostic, hint, llm_verifier, solver, verification
from agents import prompts as P
from agents.base import ask_json
from verifier import check_steps, compute_answer
from verifier.steps import normalise_step

SYSTEMS = ["B1", "B2", "B3", "B4", "OURS"]
ABLATIONS = ["A1", "A2", "A3", "A4"]
STUDENT_ONLY = ["DET"]
HINT_RUNS_D1 = {1}


@dataclass(frozen=True)
class Options:
    verify_mode: str = "steps"
    use_diagnostic: bool = True
    graded_hints: bool = True


_ABLATION_OPTIONS = {
    "A2": Options(verify_mode="final"),
    "A3": Options(use_diagnostic=False),
    "A4": Options(graded_hints=False),
}


def hints_wanted(problem: dict) -> bool:
    """Hints on D1 are made in run 1 only. Student work always gets hints."""
    if problem.get("dataset") == "d1":
        return llm.current_run() in HINT_RUNS_D1
    return True


def _outcome(status, solution=None, hints=None, **extra) -> dict:
    solution = solution if isinstance(solution, dict) else {}
    out = {
        "status": status,
        "answer": solution.get("answer"),
        "trace": solution.get("steps"),
        "hints": hints or [],
        "self_check": solution.get("self_check"),
        "retries": 0,
        "verify_reasons": [],
        "unchecked_steps": None,
        "weak_steps": None,
        "pred_first_wrong_step": None,
        "pred_misconception": None,
        "step_flags": None,
        "answer_flag": None,
    }
    out.update(extra)
    return out


def _single_hints(reply) -> list:
    raw = reply.get("hints") if isinstance(reply, dict) else None
    hints = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            claims = item.get("claims")
            hints.append({"level": item.get("level"), "text": str(item.get("text") or ""),
                          "claims": claims if isinstance(claims, list) else [], "withheld": False,
                          "regenerations": 0})
    return hints


def _single(problem: dict, with_self_check: bool) -> dict:
    prompt = P.single_llm(problem)
    reply = ask_json("single", prompt)
    if reply is not None and with_self_check:
        checked = ask_json("self_check", P.self_check(prompt, P.compact(reply)))
        reply = checked if checked is not None else reply
    if reply is None:
        return _outcome("parse_error")
    step = reply.get("first_wrong_step")
    if not isinstance(step, int) or isinstance(step, bool):
        step = None
    return _outcome("answered", reply, _single_hints(reply), pred_first_wrong_step=step,
                    pred_misconception=reply.get("misconception"))


def _diagnose(problem: dict, taxonomy: list, solution) -> tuple:
    if not problem.get("student_steps"):
        return None, None
    found = diagnostic.diagnose(problem, taxonomy, P.compact(solution))
    return found["first_wrong_step"], found["misconception"]


def _agents_unverified(problem: dict, taxonomy: list) -> dict:
    solution = solver.solve(problem)
    if solution is None:
        return _outcome("parse_error")
    wrong_step, misconception = _diagnose(problem, taxonomy, solution)
    hints = hint.make_hints(problem, solution, misconception, wrong_step) if hints_wanted(problem) else []
    return _outcome("answered", solution, hints, pred_first_wrong_step=wrong_step, pred_misconception=misconception)


def _llm_verified(problem: dict, taxonomy: list) -> dict:
    result = verification.solve_checked(problem, lambda s: llm_verifier.check_solution(problem, s))
    if not result.verified:
        no_reply = all(r == verification.NOT_JSON for r in result.reasons)
        return _outcome("parse_error" if no_reply else "abstained", retries=result.retries,
                        verify_reasons=result.reasons)
    solution = result.solution
    wrong_step, misconception = _diagnose(problem, taxonomy, solution)

    def review(text, claims):
        return llm_verifier.check_hint(problem, solution, text, claims), {}

    hints = hint.make_hints(problem, solution, misconception, wrong_step, review=review) if hints_wanted(problem) else []
    return _outcome("answered", solution, hints, retries=result.retries, verify_reasons=result.reasons,
                    pred_first_wrong_step=wrong_step, pred_misconception=misconception)


def _student_check_steps(problem: dict) -> list:
    steps = []
    for st in problem["student_steps"]:
        step = normalise_step(st["check"]) if st.get("check") is not None else {}
        steps.append({"text": st.get("step_text", ""), **step})
    return steps


def _ours(problem: dict, taxonomy: list, options: Options) -> dict:
    result = verification.solve_flagged(problem, options.verify_mode)
    if result.solution is None:
        return _outcome("parse_error", retries=result.retries, verify_reasons=result.reasons)
    delivered = result.delivered()
    misconception = wrong_step = None
    if problem.get("student_steps"):
        reason = None
        if options.verify_mode == "steps":
            report = check_steps(problem["topic"], problem, _student_check_steps(problem))
            if report.first_bad is not None:
                wrong_step = problem["student_steps"][report.first_bad]["step_no"]
                reason = f"step {wrong_step}: {report.reason}"
        if options.use_diagnostic and (wrong_step is not None or options.verify_mode != "steps"):
            found = diagnostic.diagnose(problem, taxonomy, P.compact(delivered), reason)
            misconception = found["misconception"]
            if options.verify_mode != "steps":
                wrong_step = found["first_wrong_step"]
    hints = []
    if hints_wanted(problem):
        certified = delivered["answer"] if result.answer_flag == verification.CERTIFIED else None
        review = hint.certificate_review(problem, [certified, compute_answer(problem["topic"], problem)])
        hints = hint.make_hints(problem, delivered, misconception, wrong_step, review=review,
                                graded=options.graded_hints)
    status = "answer_withheld" if result.answer_flag == verification.WITHHELD else "answered"
    return _outcome(status, result.solution, hints, retries=result.retries, verify_reasons=result.reasons,
                    step_flags=result.step_flags, answer_flag=result.answer_flag,
                    unchecked_steps=result.step_flags.count(verification.UNVERIFIED),
                    pred_first_wrong_step=wrong_step, pred_misconception=misconception)


def _detector(problem: dict) -> dict:
    reply = ask_json("detector", P.detector(problem))
    if reply is None:
        return _outcome("parse_error")
    step = reply.get("first_wrong_step")
    if not isinstance(step, int) or isinstance(step, bool):
        step = None
    return _outcome("answered", pred_first_wrong_step=step)


def run_system(system: str, problem: dict, taxonomy: list) -> dict:
    if system == "B1":
        return _single(problem, with_self_check=False)
    if system == "B2":
        return _llm_verified(problem, taxonomy)
    if system == "B3":
        return _single(problem, with_self_check=True)
    if system in ("B4", "A1"):
        return _agents_unverified(problem, taxonomy)
    if system == "OURS":
        return _ours(problem, taxonomy, Options())
    if system in _ABLATION_OPTIONS:
        return _ours(problem, taxonomy, _ABLATION_OPTIONS[system])
    if system == "DET":
        return _detector(problem)
    raise ValueError(f"unknown system {system!r}")
