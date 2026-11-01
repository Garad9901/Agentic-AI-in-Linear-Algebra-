"""Systems under test (PREREGISTRATION.md section 5).

D1 (worked solutions, RQ1/RQ4):  S0, B1, B2, B3, A1, A2
D3 (student work, RQ2/RQ3):      S0, B1, B4, A3, A4

Each run function returns one JSON-safe record. Records hold what each system
produced and what it delivered to the student; they never contain grades. Grading
against the stored solution happens later in scripts/analyse.py (circularity guard).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import prompts as P
from .leakage import detect
from .llm import LLMError, extract_json
from .verifier import Verifier

D1_SYSTEMS = ["S0", "B1", "B2", "B3", "A1", "A2"]
D3_SYSTEMS = ["S0", "B1", "B4", "A3", "A4"]


@dataclass
class Ctx:
    llm: object
    seed: int
    max_repairs: int = 2          # repair rounds after a refuted claim (fixed before test runs)
    leak_guard: bool = True       # S0 hint guard: reject hints the Verification Agent sees as leaking
    taxonomy: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def ask(self, system: str, user: str):
        """One LLM call. Returns (parsed JSON or None). One re-ask on unparseable output."""
        role = system.split("\n", 1)[0].replace("ROLE: ", "")
        for attempt in range(2):
            u = user if attempt == 0 else user + "\n\nYour last reply was not valid JSON. Reply with one JSON object only."
            rec = {"role": role, "reask": attempt}
            try:
                r = self.llm.chat(system, u, seed=self.seed)
            except LLMError as e:
                rec.update(error=str(e)[:300])
                self.calls.append(rec)
                return None
            rec.update(latency_s=round(r.latency_s, 3), prompt_tokens=r.prompt_tokens,
                       completion_tokens=r.completion_tokens, http_attempts=r.attempts,
                       rate_limit_errors=r.rate_limit_errors, run_mode=r.run_mode)
            try:
                out = extract_json(r.text)
                rec["parse_ok"] = True
                self.calls.append(rec)
                return out
            except ValueError:
                rec.update(parse_ok=False, reply_head=(r.text or "")[:200])
                self.calls.append(rec)
        return None


# ---------------------------------------------------------------- helpers
def flatten(sol) -> list[dict]:
    """Solution JSON -> list of {"claim", "step", "final"}. Malformed parts are skipped."""
    items = []
    if not isinstance(sol, dict):
        return items
    for i, st in enumerate(sol.get("steps") or [], 1):
        if isinstance(st, dict):
            for c in st.get("claims") or []:
                if isinstance(c, dict):
                    items.append({"claim": c, "step": i, "final": False})
    for c in sol.get("final_answer") or []:
        if isinstance(c, dict):
            items.append({"claim": c, "step": None, "final": True})
    return items


def _feedback(items, verdicts):
    lines = []
    for it, v in zip(items, verdicts):
        where = f"step {it['step']}" if it["step"] else "final answer"
        lines.append(f"- {where}: {json.dumps(it['claim'], separators=(',', ':'))}  -> {v}")
    return "\n".join(lines)


def _base(system, problem, ctx):
    m = ctx.llm
    return {"system": system, "dataset": problem.get("dataset", "D1"), "problem_id": problem["id"],
            "topic": problem["topic"], "size": problem["size"], "split": problem.get("split"),
            "seed": ctx.seed, "provider": getattr(m, "provider", None), "model_id": getattr(m, "model_id", None),
            "prompt_version": P.PROMPT_VERSION, "max_repairs": ctx.max_repairs}


def _finish(rec, ctx, t0):
    rec["calls"] = ctx.calls
    rec["n_calls"] = len(ctx.calls)
    rec["latency_s"] = round(time.time() - t0, 3)
    modes = {c.get("run_mode") for c in ctx.calls if c.get("run_mode")}
    rec["run_mode"] = "DRYRUN" if "DRYRUN" in modes else "LIVE"
    rec["api_errors"] = sum(1 for c in ctx.calls if "error" in c)
    return rec


# ---------------------------------------------------------------- D1
def run_d1(system: str, problem: dict, ctx: Ctx) -> dict:
    t0 = time.time()
    rec = _base(system, problem, ctx)
    pb = P.problem_block(problem)
    req = P.FINAL_REQUIRED[problem["topic"]]
    ver = Verifier(problem)
    rounds = []

    if system in ("S0", "A2", "A1"):
        sol = ctx.ask(P.SOLVER_SYS, P.SOLVER_USER.format(problem=pb, required=req))
        if system == "A1":
            items = flatten(sol)
            delivered = [dict(it, flag="unchecked") for it in items]
            withheld = []
        else:
            final_only = system == "A2"
            for rnd in range(ctx.max_repairs + 1):
                items = flatten(sol)
                idx = [i for i, it in enumerate(items) if it["final"] or not final_only]
                checked = [items[i] for i in idx]
                verdicts = [ver.check(it["claim"]).to_dict() for it in checked]
                rounds.append({"round": rnd, "n_claims": len(items), "checked": len(checked),
                               "verdicts": verdicts, "checked_index": idx})
                bad = [(it, v) for it, v in zip(checked, verdicts) if v["status"] == "refuted"]
                if not bad or rnd == ctx.max_repairs:
                    break
                new = ctx.ask(P.SOLVER_SYS, P.REPAIR_USER.format(
                    problem=pb, required=req,
                    feedback=_feedback([b[0] for b in bad], [b[1]["reason"] for b in bad])))
                if new is None:
                    break
                sol = new
            items = flatten(sol)
            # recompute on the delivered version (it may be the last repaired one)
            status = {}
            for it_idx, it in enumerate(items):
                if final_only and not it["final"]:
                    status[it_idx] = "unchecked"
                else:
                    status[it_idx] = ver.check(it["claim"]).status
            delivered, withheld = [], []
            for i, it in enumerate(items):
                s = status[i]
                if s == "refuted":
                    withheld.append(dict(it, flag="refuted"))
                else:
                    flag = {"certified": "certified", "uncheckable": "unverified"}.get(s, s)
                    delivered.append(dict(it, flag=flag))
            rec["final_verdicts"] = [status[i] for i in range(len(items))]

    elif system == "B1":
        sol = ctx.ask(P.SINGLE_TUTOR_SYS, P.SOLVER_USER.format(problem=pb, required=req))
        items = flatten(sol)
        delivered, withheld = [dict(it, flag="unchecked") for it in items], []

    elif system == "B3":
        first = ctx.ask(P.SINGLE_TUTOR_SYS, P.SOLVER_USER.format(problem=pb, required=req))
        sol = first
        if first is not None:
            second = ctx.ask(P.SINGLE_TUTOR_SYS, P.SELF_VERIFY_USER.format(
                problem=pb, required=req, previous=json.dumps(first, separators=(",", ":"))))
            if second is not None:
                sol = second
        rec["first_pass_claims"] = flatten(first)
        items = flatten(sol)
        delivered, withheld = [dict(it, flag="unchecked") for it in items], []

    elif system == "B2":
        sol = ctx.ask(P.SOLVER_SYS, P.SOLVER_USER.format(problem=pb, required=req))
        judg = {}
        for rnd in range(ctx.max_repairs + 1):
            items = flatten(sol)
            claims_txt = "\n".join(f"[{i}] {json.dumps(it['claim'], separators=(',', ':'))}"
                                   for i, it in enumerate(items))
            jr = ctx.ask(P.LLM_VERIFIER_SYS, P.LLM_VERIFIER_USER.format(problem=pb, claims=claims_txt)) or {}
            judg = {}
            for j in jr.get("judgements") or []:
                try:
                    judg[int(j["index"])] = (str(j.get("verdict", "")).lower(), str(j.get("reason", ""))[:300])
                except (KeyError, ValueError, TypeError):
                    pass
            bad = [(it, judg[i][1]) for i, it in enumerate(items) if judg.get(i, ("correct",))[0] == "incorrect"]
            rounds.append({"round": rnd, "n_claims": len(items),
                           "judgements": {str(k): v[0] for k, v in judg.items()}})
            if not bad or rnd == ctx.max_repairs:
                break
            new = ctx.ask(P.SOLVER_SYS, P.REPAIR_USER.format(
                problem=pb, required=req, feedback=_feedback([b[0] for b in bad], [b[1] for b in bad])))
            if new is None:
                break
            sol = new
        items = flatten(sol)
        delivered, withheld = [], []
        for i, it in enumerate(items):
            v = judg.get(i, ("correct",))[0]
            (withheld if v == "incorrect" else delivered).append(
                dict(it, flag="llm_rejected" if v == "incorrect" else "llm_approved"))
    else:
        raise ValueError(f"{system} is not a D1 system")

    rec.update(rounds=rounds, delivered=delivered, withheld=withheld, parsed=sol is not None)
    return _finish(rec, ctx, t0)


# ---------------------------------------------------------------- D3
def _step_claims(solution):
    out = []
    for st in solution["steps"]:
        c = st.get("claim")
        out.append(c if isinstance(c, dict) else {"type": "statement", "text": st.get("text", "")})
    return out


def _taxonomy_txt(ctx):
    return "\n".join(f"- {t['label']}: {t.get('definition', '')}" for t in ctx.taxonomy) or "- (taxonomy not set)"


def run_d3(system: str, problem: dict, solution: dict, ctx: Ctx) -> dict:
    t0 = time.time()
    rec = _base(system, problem, ctx)
    rec.update(dataset="D3", solution_id=solution["solution_id"], student_id=solution["student_id"],
               split=solution.get("split"))
    sb = P.student_block(problem, solution)

    if system == "B4":
        out = ctx.ask(P.DETECTOR_SYS, P.DETECTOR_USER.format(student=sb)) or {}
        rec["predicted_first_error"] = _step_no(out.get("first_error_step"))
        return _finish(rec, ctx, t0)

    if system == "B1":
        out = ctx.ask(P.SINGLE_TUTOR_STUDENT_SYS,
                      P.SINGLE_TUTOR_STUDENT_USER.format(student=sb, taxonomy=_taxonomy_txt(ctx))) or {}
        rec["predicted_first_error"] = _step_no(out.get("first_error_step"))
        rec["misconception"] = out.get("misconception")
        rec["hints"] = [h for h in (out.get("hints") or []) if isinstance(h, dict)]
        rec["hints_delivered"] = rec["hints"]
        return _finish(rec, ctx, t0)

    if system not in ("S0", "A3", "A4"):
        raise ValueError(f"{system} is not a D3 system")

    ver = Verifier(problem)
    vr = ver.check_steps(_step_claims(solution))
    rec["step_check"] = vr
    first = vr["first_error_index"]
    rec["predicted_first_error"] = None if first is None else first + 1
    if first is None:
        rec["misconception"], rec["hints"], rec["hints_delivered"] = None, [], []
        return _finish(rec, ctx, t0)
    reason = vr["verdicts"][first]["reason"]

    if system == "A3":
        misc = None
    else:
        d = ctx.ask(P.DIAGNOSTIC_SYS, P.DIAGNOSTIC_USER.format(
            student=sb, step=first + 1, reason=reason, taxonomy=_taxonomy_txt(ctx))) or {}
        misc = d.get("misconception")
    rec["misconception"] = misc

    hint_sys = P.HINT_SINGLE_SYS if system == "A4" else P.HINT_SYS
    user = P.HINT_USER.format(student=sb, step=first + 1, reason=reason,
                              misconception=misc if misc else "(not diagnosed)")
    answer = ver.answer_claims()

    def vjudge(c):
        return ver.check(c).status == "certified"

    hints, guard_log, why = [], [], []
    for rnd in range(ctx.max_repairs + 1):
        out = ctx.ask(hint_sys, user if rnd == 0 else user + P.HINT_RETRY_SUFFIX.format(why="; ".join(why)))
        hints = [h for h in ((out or {}).get("hints") or []) if isinstance(h, dict)]
        why = []
        for h in hints:
            for c in h.get("claims") or []:
                if isinstance(c, dict) and ver.check(c).status == "refuted":
                    why.append("a hint states a value that is mathematically wrong")
                    break
        if ctx.leak_guard and detect(hints, problem["topic"], answer, vjudge)["leak"]:
            why.append("a hint gives away the final answer")
        guard_log.append({"round": rnd, "n_hints": len(hints), "rejected_because": why})
        if not why:
            break
    rec["hints"] = hints
    # deliver only hints whose claims are not refuted and that do not leak (per the Verification Agent)
    deliver = []
    for h in hints:
        bad = any(isinstance(c, dict) and ver.check(c).status == "refuted" for c in h.get("claims") or [])
        leak = ctx.leak_guard and detect([h], problem["topic"], answer, vjudge)["leak"]
        if not bad and not leak:
            deliver.append(h)
    rec["hints_delivered"] = deliver
    rec["guard_log"] = guard_log
    return _finish(rec, ctx, t0)


def _step_no(x):
    if x is None or x == "null":
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
