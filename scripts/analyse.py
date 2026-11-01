"""Grade run records against the stored ground truth and build the paper tables T2-T7.

  python scripts/analyse.py --split test            # paper tables from LIVE records
  python scripts/analyse.py --split dev --allow-dryrun --root results/dryrun   # plumbing check only

Grading uses latutor.grader (stored solution) only; the Verification Agent is never used here.
Records tagged DRYRUN are refused unless --allow-dryrun, and then every output file is prefixed
DRYRUN_NOT_RESULTS_ so it cannot be mistaken for a result.

Paired tests over 3 seeds: the pre-registration does not say how the 3 runs enter McNemar.
This script reports
  * majority  - an item counts as an event if it occurs in >= 2 of the 3 runs (primary here), and
  * per_seed  - one McNemar per seed (sensitivity).
Log the choice in docs/DEVIATIONS.md BEFORE the first test run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latutor.generator import load  # noqa: E402
from latutor.grader import Grader, gold_final_claims  # noqa: E402
from latutor.leakage import detect  # noqa: E402
from latutor.stats import cluster_bootstrap, cohen_kappa, holm, mcnemar, mean_sd  # noqa: E402

REQUIRED_TYPES = {"rref": {"rref", "rank", "pivot_columns"}, "det": {"determinant"}, "inverse": {"inverse"},
                  "eigen": {"eigenvalues", "characteristic_polynomial", "eigenspace_basis"},
                  "diag": {"eigenvalues", "eigenspace_basis", "diagonalisable", "diagonalisation"},
                  "lsq": {"normal_equations", "lsq_solution"}}


def read_records(root: Path, split: str, dataset: str):
    out = []
    for f in sorted((root / split / dataset).glob("*/*/seed*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r["_model"] = f.parts[-3]
                out.append(r)
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def fmt(m, sd):
    return f"{m:.3f} ± {sd:.3f}" if m == m else "n/a"


# ------------------------------------------------------------------ D1
def grade_d1(recs, problems):
    graded = []
    for r in recs:
        p = problems[r["problem_id"]]
        g = Grader(p)
        grades = [g.grade(it["claim"]) for it in r["delivered"]]
        delivered_final = [(it, gr) for it, gr in zip(r["delivered"], grades) if it["final"]]
        types_ok = {it["claim"].get("type") for it, gr in delivered_final if gr == "correct"}
        graded.append({
            "model": r["_model"], "system": r["system"], "problem_id": r["problem_id"], "topic": r["topic"],
            "size": r["size"], "seed": r["seed_index"],
            "wrong": int(any(gr == "wrong" for gr in grades)),
            "n_delivered": len(grades), "n_wrong": grades.count("wrong"), "n_unscorable": grades.count("unscorable"),
            "n_withheld": len(r.get("withheld", [])),
            "final_complete_correct": int(REQUIRED_TYPES[r["topic"]] <= types_ok
                                          and not any(gr == "wrong" for _, gr in delivered_final)),
            "no_final_delivered": int(not delivered_final),
            "parsed": int(r.get("parsed", True)),
            "final_verdicts": r.get("final_verdicts"),
            "n_calls": r["n_calls"], "latency_s": r["latency_s"],
            "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in r["calls"]),
            "rate_limit_errors": sum(c.get("rate_limit_errors") or 0 for c in r["calls"]),
            "api_errors": r.get("api_errors", 0),
        })
    return graded


def rate_table(rows, keys, metric):
    """mean ± SD over seeds of the per-seed rate, range, and cluster-bootstrap CI (cluster = problem)."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        by_seed = defaultdict(list)
        for r in rs:
            by_seed[r["seed"]].append(r[metric])
        rates = [sum(v) / len(v) for v in by_seed.values()]
        m, sd = mean_sd(rates)
        _, lo, hi = cluster_bootstrap([r[metric] for r in rs], [r.get("cluster", r.get("problem_id")) for r in rs])
        row = dict(zip(keys, k))
        row.update({"n_items": len({r.get("cluster", r.get("problem_id")) for r in rs}), "runs": len(by_seed),
                    f"{metric}_mean": round(m, 4), f"{metric}_sd": round(sd, 4),
                    "min_run": round(min(rates), 4), "max_run": round(max(rates), 4),
                    "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "mean_±_sd": fmt(m, sd)})
        out.append(row)
    return out


def paired(rows, sys_a, sys_b, metric, item_key="problem_id"):
    """Returns McNemar results (majority and per-seed) for sys_a vs sys_b on metric (1 = event)."""
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["system"], r["seed"])][r[item_key]] = r[metric]
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {}
    for s in seeds:
        a, b = idx.get((sys_a, s), {}), idx.get((sys_b, s), {})
        common = sorted(set(a) & set(b))
        if common:
            per_seed[s] = mcnemar([a[i] for i in common], [b[i] for i in common])
    items = sorted(set.intersection(*[set(idx.get((sy, s), {})) for sy in (sys_a, sys_b) for s in seeds])) \
        if seeds else []
    maj = None
    if items:
        need = len(seeds) // 2 + 1
        xa = [int(sum(idx[(sys_a, s)][i] for s in seeds) >= need) for i in items]
        xb = [int(sum(idx[(sys_b, s)][i] for s in seeds) >= need) for i in items]
        maj = mcnemar(xa, xb)
    return maj, per_seed


def analyse_d1(recs, problems, out, prefix):
    graded = grade_d1(recs, problems)
    write_csv(out / f"{prefix}d1_graded_responses.csv", [{k: v for k, v in g.items() if k != "final_verdicts"}
                                                         for g in graded])
    for g in graded:
        g["cluster"] = g["problem_id"]
    # T2 wrong-claim rate (RQ1)
    t2 = rate_table(graded, ["model", "system"], "wrong") + \
        rate_table(graded, ["model", "system", "topic", "size"], "wrong")
    write_csv(out / f"{prefix}T2_wrong_claim_rate.csv", t2)
    write_csv(out / f"{prefix}T2b_final_answer_complete_and_correct.csv",
              rate_table(graded, ["model", "system"], "final_complete_correct"))
    # T3 coverage (RQ4): certified / all extracted claims in S0's delivered version
    cov = []
    for g in graded:
        if g["system"] == "S0" and g["final_verdicts"]:
            v = g["final_verdicts"]
            cov.append({**g, "certified_share": sum(x == "certified" for x in v) / len(v),
                        "decided_share": sum(x in ("certified", "refuted") for x in v) / len(v),
                        "n_claims": len(v)})
    t3 = []
    groups = defaultdict(list)
    for c in cov:
        groups[(c["model"], c["topic"], c["size"])].append(c)
        groups[(c["model"], "all", "all")].append(c)
    for k, cs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        # claim-level ratio, bootstrap clustered by problem
        num_ = [c["certified_share"] * c["n_claims"] for c in cs]
        den = [c["n_claims"] for c in cs]
        ratio = sum(num_) / sum(den)
        pairs = list(zip(num_, den))
        _, lo, hi = cluster_bootstrap(pairs, [c["problem_id"] for c in cs],
                                      stat=lambda v: sum(a for a, _ in v) / sum(b for _, b in v))
        decided = sum(c["decided_share"] * c["n_claims"] for c in cs) / sum(den)
        t3.append({"model": k[0], "topic": k[1], "size": k[2], "claims": sum(den),
                   "certified_coverage": round(ratio, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                   "decidable_share_certified_or_refuted": round(decided, 4)})
    write_csv(out / f"{prefix}T3_coverage_claims.csv", t3)
    # T5 ablations (D1 part)
    write_csv(out / f"{prefix}T5_ablations_d1.csv",
              rate_table([g for g in graded if g["system"] in ("S0", "A1", "A2")], ["model", "system"], "wrong"))
    # T7 cost
    t7 = []
    groups = defaultdict(list)
    for g in graded:
        groups[(g["model"], g["system"])].append(g)
    for (m, s), gs in sorted(groups.items()):
        t7.append({"model": m, "system": s, "responses": len(gs),
                   "mean_calls": round(sum(g["n_calls"] for g in gs) / len(gs), 2),
                   "mean_latency_s": round(sum(g["latency_s"] for g in gs) / len(gs), 2),
                   "mean_tokens": round(sum(g["tokens"] for g in gs) / len(gs), 1),
                   "total_calls": sum(g["n_calls"] for g in gs),
                   "rate_limit_errors": sum(g["rate_limit_errors"] for g in gs),
                   "api_errors": sum(g["api_errors"] for g in gs),
                   "unparsed_responses": sum(1 - g["parsed"] for g in gs)})
    write_csv(out / f"{prefix}T7_cost_d1.csv", t7)
    # primary tests RQ1
    tests = []
    for m in sorted({g["model"] for g in graded}):
        rows = [g for g in graded if g["model"] == m]
        for base in ("B1", "B2", "B3"):
            maj, per = paired(rows, "S0", base, "wrong")
            if maj:
                tests.append({"model": m, "rq": "RQ1", "comparison": f"S0 vs {base}", "pairing": "majority", **maj})
            for s, res in per.items():
                tests.append({"model": m, "rq": "RQ1", "comparison": f"S0 vs {base}", "pairing": f"seed{s}", **res})
        for abl in ("A1", "A2"):
            maj, _ = paired(rows, "S0", abl, "wrong")
            if maj:
                tests.append({"model": m, "rq": "RQ1-ablation (exploratory)", "comparison": f"S0 vs {abl}",
                              "pairing": "majority", **maj})
    return graded, tests


# ------------------------------------------------------------------ D3
def analyse_d3(recs, d3_dir, out, prefix):
    probs = {p["id"]: p for p in load(d3_dir / "problems.jsonl")}
    sols = {s["solution_id"]: s for s in load(d3_dir / "solutions.jsonl")}
    rows = []
    for r in recs:
        s, p = sols[r["solution_id"]], probs[r["problem_id"]]
        gold = s["gold"]
        g_err = gold.get("first_error_step")
        pred = r.get("predicted_first_error")
        g = Grader(p)
        hints = r.get("hints_delivered") or []
        lk = detect(hints, p["topic"], gold_final_claims(p), lambda c: g.grade(c) == "correct") if hints else \
            {"leak": False, "per_hint": []}
        rows.append({"model": r["_model"], "system": r["system"], "solution_id": r["solution_id"],
                     "student_id": r["student_id"], "cluster": r["student_id"], "topic": p["topic"],
                     "seed": r["seed_index"], "gold_has_error": int(g_err is not None),
                     "pred_has_error": int(pred is not None),
                     "exact_first_error": int(g_err is not None and pred == g_err),
                     "gold_label": gold.get("misconception"), "pred_label": r.get("misconception"),
                     "n_hints": len(hints), "leak": int(lk["leak"]),
                     "hint_leak_count": sum(lk["per_hint"]),
                     "n_calls": r["n_calls"], "latency_s": r["latency_s"],
                     "steps_total": len(s["steps"]),
                     "steps_certified": (r.get("step_check") or {}).get("n_certified"),
                     "steps_refuted": (r.get("step_check") or {}).get("n_refuted")})
    write_csv(out / f"{prefix}d3_graded.csv", rows)
    err_rows = [x for x in rows if x["gold_has_error"]]
    t4 = rate_table([{**x, "problem_id": x["solution_id"]} for x in err_rows], ["model", "system"], "exact_first_error")
    # detection P/R/F1 and misconception macro-F1
    extra = []
    for (m, sy) in sorted({(x["model"], x["system"]) for x in rows}):
        rs = [x for x in rows if x["model"] == m and x["system"] == sy]
        tp = sum(x["gold_has_error"] and x["pred_has_error"] for x in rs)
        fp = sum((not x["gold_has_error"]) and x["pred_has_error"] for x in rs)
        fn = sum(x["gold_has_error"] and not x["pred_has_error"] for x in rs)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
        labs = [x for x in rs if x["gold_has_error"] and x["pred_label"] is not None]
        macro = float("nan")
        if labs:
            cats = sorted({x["gold_label"] for x in labs})
            f1s = []
            for c in cats:
                ctp = sum(x["gold_label"] == c and x["pred_label"] == c for x in labs)
                cfp = sum(x["gold_label"] != c and x["pred_label"] == c for x in labs)
                cfn = sum(x["gold_label"] == c and x["pred_label"] != c for x in labs)
                f1s.append(2 * ctp / (2 * ctp + cfp + cfn) if ctp + cfp + cfn else 0.0)
            macro = sum(f1s) / len(f1s)
        extra.append({"model": m, "system": sy, "detect_precision": round(prec, 4), "detect_recall": round(rec, 4),
                      "detect_f1": round(f1, 4), "misconception_macro_f1": round(macro, 4)})
    write_csv(out / f"{prefix}T4_first_error.csv", t4)
    write_csv(out / f"{prefix}T4b_detection_and_labels.csv", extra)
    hint_rows = [{**x, "problem_id": x["solution_id"]} for x in rows if x["system"] in ("S0", "B1", "A3", "A4")
                 and x["gold_has_error"]]
    write_csv(out / f"{prefix}T6_hint_leakage_auto.csv", rate_table(hint_rows, ["model", "system"], "leak"))
    # step coverage (RQ4, student steps)
    cov = [x for x in rows if x["system"] == "S0" and x["steps_certified"] is not None]
    if cov:
        write_csv(out / f"{prefix}T3b_coverage_student_steps.csv", [{
            "model": m, "steps": sum(x["steps_total"] for x in cov if x["model"] == m),
            "certified_or_refuted_share": round(sum(x["steps_certified"] + x["steps_refuted"] for x in cov
                                                    if x["model"] == m)
                                                / max(1, sum(x["steps_total"] for x in cov if x["model"] == m)), 4)}
            for m in sorted({x["model"] for x in cov})])
    # inter-annotator agreement (pre-discussion), test split
    a1 = [s["gold"].get("annotator1", {}) for s in sols.values() if s.get("split") == "test"]
    a2 = [s["gold"].get("annotator2", {}) for s in sols.values() if s.get("split") == "test"]
    if a1 and all(a1) and all(a2):
        k_step = cohen_kappa([str(x.get("first_error_step")) for x in a1], [str(x.get("first_error_step")) for x in a2])
        k_lab = cohen_kappa([str(x.get("misconception")) for x in a1], [str(x.get("misconception")) for x in a2])
        write_csv(out / f"{prefix}T1b_kappa.csv", [{"kappa_first_error_step": round(k_step, 4),
                                                   "kappa_misconception": round(k_lab, 4), "n": len(a1)}])
    tests = []
    for m in sorted({x["model"] for x in rows}):
        er = [{**x, "problem_id": x["solution_id"]} for x in err_rows if x["model"] == m]
        maj, per = paired([{**x, "miss": 1 - x["exact_first_error"]} for x in er], "S0", "B4", "miss")
        if maj:
            tests.append({"model": m, "rq": "RQ2", "comparison": "S0 vs B4 (event = first error missed)",
                          "pairing": "majority", **maj})
        for s, res in per.items():
            tests.append({"model": m, "rq": "RQ2", "comparison": "S0 vs B4 (event = first error missed)",
                          "pairing": f"seed{s}", **res})
        hr = [x for x in hint_rows if x["model"] == m]
        maj, _ = paired(hr, "S0", "B1", "leak")
        if maj:
            tests.append({"model": m, "rq": "RQ3 (automated leakage)", "comparison": "S0 vs B1",
                          "pairing": "majority", **maj})
    return rows, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default="results")
    ap.add_argument("--d1", default="data/d1/problems.jsonl")
    ap.add_argument("--d3", default="data/private/d3")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--allow-dryrun", action="store_true")
    args = ap.parse_args()
    root, out = ROOT / args.root, ROOT / args.out
    d1 = read_records(root, args.split, "D1")
    d3 = read_records(root, args.split, "D3")
    modes = {r.get("run_mode") for r in d1 + d3}
    prefix = ""
    if "DRYRUN" in modes:
        if not args.allow_dryrun:
            sys.exit("Refused: DRYRUN records found. These are not results.")
        prefix = "DRYRUN_NOT_RESULTS_"
        out = ROOT / "results/dryrun_tables"
    tests = []
    if d1:
        problems = {p["id"]: p for p in load(ROOT / args.d1)}
        _, t = analyse_d1(d1, problems, out, prefix)
        tests += t
    if d3:
        _, t = analyse_d3(d3, ROOT / args.d3, out, prefix)
        tests += t
    # Holm across the pre-registered primary family (per model, majority pairing)
    for m in sorted({t["model"] for t in tests}):
        fam = {t["comparison"]: t["p"] for t in tests if t["model"] == m and t["pairing"] == "majority"
               and t["rq"] in ("RQ1", "RQ2")}
        adj = holm(fam)
        for t in tests:
            if t["model"] == m and t["pairing"] == "majority" and t["comparison"] in adj and t["rq"] in ("RQ1", "RQ2"):
                t["p_holm"] = adj[t["comparison"]]
    write_csv(out / f"{prefix}tests.csv", tests)
    print(f"wrote tables to {out}  ({len(d1)} D1 records, {len(d3)} D3 records)")
    if len({r.get('seed_index') for r in d1 + d3}) < 3 and not prefix:
        print("WARNING: fewer than 3 runs per configuration; the plan requires 3.")


if __name__ == "__main__":
    main()
