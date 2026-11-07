"""Score result files into tables T1 to T7, as CSV and markdown.

  python score.py --split dev

Writes results/tables/<split>/<provider>__<model>/T*.csv and tables.md, one folder per
model. A cell that cannot be computed stays blank, and the note under the table says
why. No number here comes from anywhere but a result row or a dataset file.

The pre-registered metrics are RQ1 wrong-claim rate (T2b), RQ4 coverage (T3d) and the
McNemar tests with Holm correction across the 4 registered comparisons (T8). They use
latutor/stats.py and the per-item majority rule from docs/DEVIATIONS.md.

All reported results come from one model, the Groq one (decision of 2026-11-07). Results
from any other provider are not scored. Claims are graded by grader.py against the
ground truth; RQ4 asks the verifier what it can certify, which is a separate question.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import d3 as d3_data
import grader
from data import gen_d1, gen_d2
from extraction import extract
from latutor.stats import cluster_bootstrap, holm, mcnemar, mean_sd
from run import RESULTS, RUNS
from systems import ABLATIONS, SYSTEMS
from verifier import check_claim, leak_check, step_statuses, verify, verify_hint

TOPICS = gen_d1.TOPICS
SIZES = gen_d1.SIZES
S0 = "OURS"  # S0 in prereg-v1
RQ1_BASELINES = ["B1", "B2", "B3"]
RQ2_BASELINE = "DET"  # B4 in prereg-v1
MIN_RUNS_AGREEING = 2  # per-item majority: an event in >= 2 of 3 runs
BOOTSTRAP = 10_000
TRUNCATION_FLAG = 0.01  # above this, truncation may be distorting results


# ---------------------------------------------------------------- metrics
def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters. None if undefined (no items, or chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    if not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def f1(tp: int, fp: int, fn: int) -> float | None:
    if 2 * tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def macro_f1(gold: list, pred: list) -> float | None:
    """Macro-F1 over the labels present in gold."""
    labels = sorted(set(gold))
    if not labels:
        return None
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores.append(f1(tp, fp, fn) or 0.0)
    return sum(scores) / len(scores)


def step_counts(gold_labels: dict, predicted_first) -> tuple[int, int, int]:
    """(tp, fp, fn) for the class "wrong". Steps after the predicted first error are not
    scored, because every system predicts only the first error."""
    tp = fp = fn = 0
    for step, label in gold_labels.items():
        if predicted_first is not None and step > predicted_first:
            continue
        pred_wrong = step == predicted_first
        gold_wrong = label == "wrong"
        tp += pred_wrong and gold_wrong
        fp += pred_wrong and not gold_wrong
        fn += gold_wrong and not pred_wrong
    return tp, fp, fn


# ---------------------------------------------------------------- loading
def parse_name(path: Path, split: str):
    """<system>_<dataset>_<split>_run<k>__<provider>__<model>.csv -> (system, dataset, "provider__model")."""
    head, _, tag = path.stem.partition("__")
    parts = head.split("_")
    if not tag or len(parts) != 4 or parts[2] != split or parts[1] not in ("d1", "d2", "d3"):
        return None
    if not parts[3].startswith("run"):
        return None
    return parts[0], parts[1], tag


def load_results(split: str, results: Path = RESULTS) -> dict:
    """Keyed by (system, dataset, model tag). Runs are pooled: counts are over problem x run."""
    frames: dict = {}
    for path in sorted(results.glob(f"*_{split}_run*__*.csv")):
        parsed = parse_name(path, split)
        if parsed:
            frame = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"pred_misconception": str})
            frames.setdefault(parsed, []).append(frame)
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def delivered_hints(row) -> list:
    return [h for h in json.loads(row["hints_json"]) if not h.get("withheld") and h.get("text")]


def _final_status(problem: dict, answer) -> str:
    verdict = verify(problem["topic"], problem, answer)
    if verdict.ok:
        return "certified"
    return "uncheckable" if verdict.reason.startswith("malformed") else "refuted"


def _verifier_statuses(problem: dict, claims: list, trace, answer) -> list:
    """What the Verification Agent says of each claim (RQ4). Never used to grade."""
    steps = step_statuses(problem["topic"], problem, trace) if isinstance(trace, list) else []
    statuses = []
    for item in claims:
        if item["where"] == "step":
            statuses.append(steps[item["index"]])
        elif item["where"] == "final":
            statuses.append(_final_status(problem, answer))
        else:
            statuses.append(check_claim(problem, item["claim"]))
    return statuses


def _is_delivered(item: dict, step_flags, answer_delivered: bool) -> bool:
    if item["where"] == "step":
        return step_flags is None or step_flags[item["index"]] != "withheld"
    if item["where"] == "final":
        return answer_delivered
    return True


def claim_stats(problem: dict, delivered: bool, trace, answer, hints: list, step_flags=None,
                answer_delivered: bool = True) -> dict:
    """RQ1 and RQ4 for one response.

    RQ1 counts only what reached the student: OURS withholds refuted claims. RQ4 counts every
    claim in the final version, withheld ones included, as prereg-v1 registered it. A response
    that delivers nothing (abstained, or no usable reply) makes no claims.
    """
    claims = extract(problem, trace, answer, hints) if delivered else []
    shown = [item for item in claims if _is_delivered(item, step_flags, answer_delivered)]
    grades = [grader.grade_claim(problem, item["claim"]) for item in shown]
    statuses = _verifier_statuses(problem, claims, trace, answer)
    return {
        "n_claims": len(claims),
        "n_delivered_claims": len(shown),
        "n_withheld_claims": len(claims) - len(shown),
        "n_wrong_claims": grades.count("wrong"),
        "n_unscorable_claims": grades.count("unscorable"),
        "wrong_claim": int("wrong" in grades),
        "n_weakly_graded": grades.count("weak"),
        "n_certified": statuses.count("certified"),  # weakly checked claims never count as certified
        "n_weak": statuses.count("weak"),
        "n_refuted": statuses.count("refuted"),
    }


def hints_counted(row) -> bool:
    """D1 hints are exploratory and scored in run 1 only (decision of 2026-11-07). Single-prompt
    systems write hints in every run, so their D1 hints in runs 2 and 3 are set aside."""
    return not (str(row["problem_id"]).startswith("D1-") and int(row["run"]) != 1)


def grade_d1(frame: pd.DataFrame, problems: dict) -> pd.DataFrame:
    """Add grade columns. api_error rows are dropped: they say nothing about a system."""
    usable = frame[frame["status"] != "api_error"].copy()
    correct, wrong, consistent, fabricated, delivered, leaks, verified, with_claim, n_hints = ([] for _ in range(9))
    stats, flagged = [], []
    for _, row in usable.iterrows():
        p = problems[row["problem_id"]]
        answer = json.loads(row["answer_json"])
        has_answer = row["status"] == "answered" and answer is not None
        ok = has_answer and grader.grade(p, answer)
        flags = json.loads(row["flags_json"]) if "flags_json" in row and row["flags_json"] else None
        step_flags = flags["steps"] if flags else None
        full_trace = json.loads(row["trace_json"])
        trace = full_trace
        if step_flags is not None and isinstance(full_trace, list):  # what the student saw
            trace = [s for s, f in zip(full_trace, step_flags) if f != "withheld"]
        correct.append(ok)
        wrong.append(has_answer and not ok)
        delivered.append(has_answer)
        consistent.append(has_answer and grader.consistent_computation_error(p, answer, trace))
        fabricated.append(grader.fabricated(p, answer if has_answer else None, trace))
        hints = delivered_hints(row) if hints_counted(row) else []
        flagged.append(sum(h.get("flag") == "unverified" for h in hints))
        truth = grader.answer_of(p)
        leaks.append(sum(leak_check(h["text"], p["topic"], truth, h.get("claims")).leaked for h in hints))
        checks = [verify_hint(p, h["text"], h.get("claims")) for h in hints]
        verified.append(sum(c.verified for c in checks))
        with_claim.append(sum(c.verified and c.n_certified > 0 for c in checks))
        n_hints.append(len(hints))
        made_claims = row["status"] in ("answered", "answer_withheld")
        stats.append(claim_stats(p, made_claims, full_trace, answer, hints, step_flags, row["status"] == "answered"))
    usable["hints_flagged_unverified"] = flagged
    for column in ("n_claims", "n_delivered_claims", "n_withheld_claims", "n_wrong_claims", "n_unscorable_claims",
                   "n_weakly_graded", "wrong_claim", "n_certified", "n_weak", "n_refuted"):
        usable[column] = [s[column] for s in stats]
    usable["correct"] = correct
    usable["wrong"] = wrong
    usable["delivered"] = delivered
    usable["consistent_error"] = consistent
    usable["fabricated"] = fabricated
    usable["hints_leaked"] = leaks
    usable["hints_verified"] = verified
    usable["hints_with_certified_claim"] = with_claim
    usable["hints_delivered"] = n_hints
    return usable


# ---------------------------------------------------------------- tables
def _ratio(hit: int, n: int) -> str:
    return "" if n == 0 else f"{hit / n:.3f} ({hit}/{n})"


def _num(x, digits=3) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def _fabricated_rate(frame: pd.DataFrame) -> str:
    eligible = frame[frame["topic"].isin(grader.SURFACE_TOPICS)]
    if eligible.empty:
        return "n/a"
    return _ratio(int(eligible["fabricated"].astype(bool).sum()), len(eligible))


def rate_row(frame: pd.DataFrame, metric: str, bootstrap: int) -> dict:
    """Mean +- SD over runs, the range, and a bootstrap CI clustered by problem (prereg-v1 s6)."""
    rates = frame.groupby("run")[metric].mean().tolist()
    mean, sd = mean_sd(rates)
    _, low, high = cluster_bootstrap(frame[metric].tolist(), frame["problem_id"].tolist(), B=bootstrap)
    return {"problems": frame["problem_id"].nunique(), "runs": len(rates), "mean": _num(mean), "sd": _num(sd),
            "min_run": _num(min(rates)), "max_run": _num(max(rates)), "ci95_low": _num(low), "ci95_high": _num(high)}


def t2b_wrong_claim(graded: dict, names: list, bootstrap: int) -> pd.DataFrame:
    rows = []
    for name in names:
        f = graded[name]
        rows.append({"system": name, "topic": "all", "size": "all", **rate_row(f, "wrong_claim", bootstrap)})
        for (topic, size), cell in f.groupby(["topic", "size"], sort=True):
            rows.append({"system": name, "topic": topic, "size": size, **rate_row(cell, "wrong_claim", bootstrap)})
    return pd.DataFrame(rows)


def coverage_row(frame: pd.DataFrame, bootstrap: int) -> dict:
    claims = int(frame["n_claims"].sum())
    if claims == 0:
        return {"responses": len(frame), "claims": 0}
    pairs = list(zip(frame["n_certified"].tolist(), frame["n_claims"].tolist()))

    def share(sample):
        return sum(a for a, _ in sample) / max(sum(b for _, b in sample), 1)

    _, low, high = cluster_bootstrap(pairs, frame["problem_id"].tolist(), stat=share, B=bootstrap)
    return {"responses": len(frame), "claims": claims,
            "certified_coverage": _num(frame["n_certified"].sum() / claims),
            "ci95_low": _num(low), "ci95_high": _num(high),
            "decidable_share": _num((frame["n_certified"].sum() + frame["n_refuted"].sum()) / claims),
            "weakly_checked_share": _num(frame["n_weak"].sum() / claims)}


def t3d_coverage(frame, bootstrap: int) -> pd.DataFrame:
    columns = ["topic", "size", "responses", "claims", "certified_coverage", "ci95_low", "ci95_high",
               "decidable_share", "weakly_checked_share"]
    if frame is None:
        return pd.DataFrame(columns=columns)
    rows = [{"topic": "all", "size": "all", **coverage_row(frame, bootstrap)}]
    for (topic, size), cell in frame.groupby(["topic", "size"], sort=True):
        rows.append({"topic": topic, "size": size, **coverage_row(cell, bootstrap)})
    return pd.DataFrame(rows, columns=columns, dtype=object).fillna("")


def paired_tests(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> tuple:
    """(majority result or None, {run: result}), each from latutor.stats.mcnemar."""
    left = a.set_index(["run", "problem_id"])[metric].to_dict()
    right = b.set_index(["run", "problem_id"])[metric].to_dict()
    per_run = {}
    for run in sorted({k[0] for k in left} & {k[0] for k in right}):
        items = sorted(k[1] for k in left if k[0] == run and k in right)
        if items:
            per_run[run] = mcnemar([left[(run, i)] for i in items], [right[(run, i)] for i in items])
    problems = {k[1] for k in left}
    complete = sorted(i for i in problems if all((r, i) in left and (r, i) in right for r in RUNS))
    if not complete:
        return None, per_run
    x = [int(sum(left[(r, i)] for r in RUNS) >= MIN_RUNS_AGREEING) for i in complete]
    y = [int(sum(right[(r, i)] for r in RUNS) >= MIN_RUNS_AGREEING) for i in complete]
    return mcnemar(x, y), per_run


def _test_row(rq: str, comparison: str, pairing: str, result: dict | None, adjusted="") -> dict:
    row = {"rq": rq, "comparison": comparison, "pairing": pairing}
    if result:
        row.update(n_pairs=result["n_pairs"], s0_only=result["b"], baseline_only=result["c"], p=_num(result["p"], 4),
                   method=result["method"], odds_ratio=_num(result["odds_ratio"]),
                   or_ci_low=_num(result["or_ci_low"]), or_ci_high=_num(result["or_ci_high"]), holm_p=adjusted)
    return row


def holm_family(primary: dict, size: int = 4) -> dict:
    """Holm over a whole family. Empty unless every comparison in it has a p-value."""
    if len(primary) != size or any(r is None for r in primary.values()):
        return {}
    return {k: _num(v, 4) for k, v in holm({k: r["p"] for k, r in primary.items()}).items()}


def primary_results(graded: dict, rq2: dict) -> tuple:
    """(majority, per_run) for the 4 registered comparisons of one model."""
    majority, per_run = {}, {}
    for base in RQ1_BASELINES:
        key = ("RQ1", f"{S0} vs {base}")
        if S0 in graded and base in graded:
            majority[key], per_run[key] = paired_tests(graded[S0], graded[base], "wrong_claim")
        else:
            majority[key], per_run[key] = None, {}
    key = ("RQ2", f"{S0} vs {RQ2_BASELINE}")
    if S0 in rq2 and RQ2_BASELINE in rq2:
        majority[key], per_run[key] = paired_tests(rq2[S0], rq2[RQ2_BASELINE], "first_error_wrong")
    else:
        majority[key], per_run[key] = None, {}
    return majority, per_run


def t8_tests(graded: dict, rq2: dict) -> tuple[pd.DataFrame, list]:
    majority, per_run = primary_results(graded, rq2)
    adjusted = holm_family(majority)
    rows = []
    for key, result in majority.items():
        rows.append(_test_row(key[0], key[1], "majority", result, adjusted.get(key, "")))
        for run, res in per_run[key].items():
            rows.append(_test_row(key[0], key[1], f"run{run}", res))
    columns = ["rq", "comparison", "pairing", "n_pairs", "s0_only", "baseline_only", "p", "method", "odds_ratio",
               "or_ci_low", "or_ci_high", "holm_p"]
    notes = ["One model family only (decision of 2026-11-07), so the family is the 4 registered comparisons.",
             "Primary pairing is per-item majority: an event in at least 2 of 3 runs. It stays blank until both "
             "systems have all 3 runs. Per-run rows are the sensitivity analysis and take no correction.",
             "s0_only and baseline_only are the discordant pairs. Odds ratio is s0_only / baseline_only, with an "
             "exact 95% CI, so below 1 favours OURS."]
    if not adjusted:
        notes.append("holm_p is blank: Holm runs across all 4 registered comparisons, and at least one has no "
                     "majority p-value yet. RQ2 needs D3.")
    return pd.DataFrame(rows, columns=columns, dtype=object).fillna(""), notes


def rq2_frames(d3: dict, gold: dict) -> dict:
    """RQ2 event, on erroneous solutions: the predicted first wrong step is not the gold one."""
    frames = {}
    for name, frame in d3.items():
        erroneous = [sid for sid, g in gold.items() if g["first_error_step"] is not None]
        f = frame[(frame["status"] != "api_error") & frame["problem_id"].isin(erroneous)].copy()
        predicted = [None if pd.isna(v) else int(v) for v in f["pred_first_wrong_step"]]
        f["first_error_wrong"] = [int(p != gold[s]["first_error_step"]) for p, s in zip(predicted, f["problem_id"])]
        frames[name] = f
    return frames


def t1_dataset(problems: list[dict], d3_frame, d3_note) -> tuple[pd.DataFrame, list]:
    rows = []
    for topic in TOPICS:
        row = {"dataset": "D1", "topic": topic}
        for size in SIZES:
            row[f"size_{size}"] = sum(p["topic"] == topic and p["size"] == size for p in problems)
        row["dev"] = sum(p["topic"] == topic and p["split"] == "dev" for p in problems)
        row["test"] = sum(p["topic"] == topic and p["split"] == "test" for p in problems)
        row["total"] = sum(p["topic"] == topic for p in problems)
        rows.append(row)
    d2 = gen_d2.load()
    runnable = [p for p in d2 if p["runnable"]]
    for task, topic in gen_d2.TOPIC_OF_TASK.items():
        row = {"dataset": "D2", "topic": topic}
        for size in SIZES:
            row[f"size_{size}"] = sum(p["task"] == task and p["size"] == size for p in runnable)
        row["dev"] = sum(p["task"] == task and p["split"] == "dev" for p in runnable)
        row["test"] = sum(p["task"] == task and p["split"] == "test" for p in runnable)
        row["total"] = sum(p["task"] == task for p in runnable)
        rows.append(row)
    d3_row = {"dataset": "D3", "topic": "all"}
    notes = ["Least squares: size n means A is (n+1) x n.",
             f"D2 is the {len(runnable)} of {len(d2)} LinAlg-Bench problems that can be checked exactly. "
             "See data/SOURCES.md."]
    if d3_frame is None:
        notes.append(f"D3 row is blank: {d3_note}.")
    else:
        d3_row["total"] = d3_frame["solution_id"].nunique()
    rows.append(d3_row)
    return pd.DataFrame(rows, dtype=object).fillna(""), notes


def t2_t3_accuracy(graded: dict, by: str, values: list, names: list) -> pd.DataFrame:
    rows = []
    for value in values:
        row = {by: value}
        for name in names:
            f = graded[name]
            cell = f[f[by] == value]
            row[name] = _ratio(int(cell["correct"].sum()), len(cell))
        rows.append(row)
    total = {by: "all"}
    for name in names:
        f = graded[name]
        total[name] = _ratio(int(f["correct"].sum()), len(f))
    rows.append(total)
    return pd.DataFrame(rows)


def t3b_reliability(graded: dict, names: list, raw: dict) -> pd.DataFrame:
    rows = []
    for name in names:
        f, n = graded[name], len(graded[name])
        rows.append({
            "system": name,
            "n": n,
            "api_errors_excluded": int((raw[name]["status"] == "api_error").sum()),
            "answered": int(f["delivered"].sum()),
            "wrong_delivered_rate": _ratio(int(f["wrong"].sum()), n),
            "consistent_computation_error_rate": _ratio(int(f["consistent_error"].sum()), n),
            "fabricated_rate": _fabricated_rate(f),
            "abstention_rate": _ratio(int((f["status"] == "abstained").sum()), n),
            "answer_withheld_rate": _ratio(int((f["status"] == "answer_withheld").sum()), n),
            "no_parseable_reply": int((f["status"] == "parse_error").sum()),
            "accuracy_when_answered": _ratio(int(f["correct"].sum()), int(f["delivered"].sum())),
        })
    return pd.DataFrame(rows)


def t3c_fabricated(graded: dict, names: list) -> pd.DataFrame:
    rows = []
    for topic in TOPICS:
        row = {"topic": topic}
        for name in names:
            cell = graded[name][graded[name]["topic"] == topic]
            row[name] = _fabricated_rate(cell) if topic in grader.SURFACE_TOPICS else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def t4_student(d3_results: dict, d3_frame) -> tuple[pd.DataFrame, list]:
    columns = ["system", "solutions", "first_wrong_step_accuracy", "step_f1", "misconception_macro_f1"]
    notes = []
    if d3_frame is None:
        return pd.DataFrame([{"system": s} for s in SYSTEMS + ["DET"]], columns=columns, dtype=object).fillna(""), notes
    gold, left_out = d3_data.gold_labels(d3_frame)
    if left_out:
        notes.append(f"{left_out} solutions left out: annotators disagree and there is no gold row.")
    rows = []
    for name in sorted(d3_results):
        f = d3_results[name]
        f = f[(f["status"] != "api_error") & f["problem_id"].isin(gold)]
        hits = tp = fp = fn = 0
        gold_labels, pred_labels = [], []
        for _, row in f.iterrows():
            g = gold[row["problem_id"]]
            pred = row["pred_first_wrong_step"]
            pred = None if pd.isna(pred) else int(pred)
            hits += pred == g["first_error_step"]
            a, b, c = step_counts(g["step_labels"], pred)
            tp, fp, fn = tp + a, fp + b, fn + c
            if g["misconception"]:
                gold_labels.append(g["misconception"])
                label = row["pred_misconception"]
                pred_labels.append("" if pd.isna(label) else label)
        rows.append({"system": name, "solutions": len(f), "first_wrong_step_accuracy": _ratio(hits, len(f)),
                     "step_f1": _num(f1(tp, fp, fn)), "misconception_macro_f1": _num(macro_f1(gold_labels, pred_labels))})
    return pd.DataFrame(rows, columns=columns), notes


def t4b_kappa(d3_frame) -> tuple[pd.DataFrame, list]:
    columns = ["measure", "items", "kappa"]
    measures = ["step_label", "first_error_step", "misconception"]
    if d3_frame is None:
        return pd.DataFrame([{"measure": m} for m in measures], columns=columns, dtype=object).fillna(""), []
    names = sorted(set(d3_frame["annotator"]) - {d3_data.GOLD})
    if len(names) != 2:
        return (pd.DataFrame([{"measure": m} for m in measures], columns=columns, dtype=object).fillna(""),
                [f"Kappa needs exactly two annotators, found {len(names)}."])
    one, two = (d3_frame[d3_frame["annotator"] == name] for name in names)
    steps = one.merge(two, on=["solution_id", "step_no"], suffixes=("_1", "_2"))
    sols = steps.drop_duplicates("solution_id")
    rows = []
    for label, frame in (("step_label", steps), ("first_error_step", sols), ("misconception", sols)):
        a, b = frame[f"{label}_1"].tolist(), frame[f"{label}_2"].tolist()
        rows.append({"measure": label, "items": len(a), "kappa": _num(cohen_kappa(a, b))})
    return pd.DataFrame(rows, columns=columns), []


def t5_hints(graded: dict, names: list) -> pd.DataFrame:
    rows = []
    for name in names:
        f = graded[name]
        delivered = int(f["hints_delivered"].sum())
        withheld = sum(sum(bool(h.get("withheld")) for h in json.loads(v)) for v in f["hints_json"])
        rows.append({
            "system": name,
            "hints_delivered": delivered,
            "hints_withheld": withheld,
            "leak_rate": _ratio(int(f["hints_leaked"].sum()), delivered),
            "verified_rate": _ratio(int(f["hints_verified"].sum()), delivered),
            "verified_with_certified_claim": _ratio(int(f["hints_with_certified_claim"].sum()), delivered),
            "delivered_with_unverified_flag": _ratio(int(f["hints_flagged_unverified"].sum()), delivered),
        })
    return pd.DataFrame(rows)


def t6_ablations(graded: dict, ablation_names: list) -> pd.DataFrame:
    rows = []
    for name in ablation_names:
        row = {"configuration": name}
        if name in graded:
            f = graded[name]
            n, delivered = len(f), int(f["hints_delivered"].sum())
            row.update(n=n, accuracy=_ratio(int(f["correct"].sum()), n),
                       abstention_rate=_ratio(int((f["status"] == "abstained").sum()), n),
                       wrong_delivered_rate=_ratio(int(f["wrong"].sum()), n),
                       hint_leak_rate=_ratio(int(f["hints_leaked"].sum()), delivered))
        rows.append(row)
    columns = ["configuration", "n", "accuracy", "abstention_rate", "wrong_delivered_rate", "hint_leak_rate"]
    return pd.DataFrame(rows, columns=columns, dtype=object).fillna("")


def t7_cost(graded: dict, names: list) -> pd.DataFrame:
    rows = []
    for name in names:
        f = graded[name]
        sync = f[f["mode"] == "sync"]
        rows.append({
            "system": name,
            "problems": len(f),
            "mean_llm_calls": _num(f["llm_calls"].mean(), 2),
            "mean_input_tokens": _num(f["input_tokens"].mean(), 0),
            "mean_output_tokens": _num(f["output_tokens"].mean(), 0),
            "mean_seconds": _num((sync["llm_seconds"] + sync["local_seconds"]).mean(), 2) if len(sync) else "",
            "parse_failures": int(f["parse_failures"].sum()),
            "rate_limit_retries": int(f["rate_limit_errors"].sum()),
        })
    return pd.DataFrame(rows)


def t7b_truncation(graded: dict, names: list) -> tuple[pd.DataFrame, list]:
    rows, flagged = [], []
    for name in names:
        f = graded[name]
        for size in [*SIZES, "all"]:
            cell = f if size == "all" else f[f["size"] == size]
            calls, cut = int(cell["llm_calls"].sum()), int(cell["truncated_calls"].sum())
            rows.append({"system": name, "size": size, "calls": calls, "truncated": cut,
                         "truncation_rate": _ratio(cut, calls)})
            if calls and size != "all" and cut / calls > TRUNCATION_FLAG:
                flagged.append(f"{name} size {size}")
    notes = [f"A call is truncated if its finish_reason is length. Flag threshold: {TRUNCATION_FLAG:.0%}."]
    notes.append(f"FLAG: truncation above the threshold in {flagged}." if flagged else "No cell is above the threshold.")
    return pd.DataFrame(rows), notes


# ---------------------------------------------------------------- output
def markdown(frame: pd.DataFrame) -> str:
    def clean(x):
        return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


REPORTED_PROVIDER = "groq"


def model_tags(split: str, results: Path = RESULTS) -> list:
    """Only the reported model. The Gemini code stays, but produces no reported results."""
    return sorted({key[2] for key in load_results(split, results) if key[2].startswith(f"{REPORTED_PROVIDER}__")})


def build(split: str, tag: str | None = None, results: Path = RESULTS, bootstrap: int = BOOTSTRAP) -> list:
    problems = gen_d1.load()
    by_id = {p["id"]: p for p in problems}
    loaded = load_results(split, results)
    d1 = {name: f for (name, ds, t), f in loaded.items() if ds == "d1" and t == tag}
    d3 = {name: f for (name, ds, t), f in loaded.items() if ds == "d3" and t == tag}
    graded = {name: grade_d1(f, by_id) for name, f in d1.items()}
    base = [s for s in SYSTEMS if s in graded and not graded[s].empty]
    graded = {s: f for s, f in graded.items() if not f.empty}

    d3_frame, d3_note = None, ""
    try:
        d3_frame = d3_data.read_rows()
    except d3_data.D3Unavailable as e:
        d3_note = str(e)

    tables = []
    frame, notes = t1_dataset(problems, d3_frame, d3_note)
    tables.append(("T1", "Dataset statistics", frame, notes))
    missing = [s for s in SYSTEMS if s not in graded]
    d1_notes = [f"Accuracy is correct / all problems, so an abstention counts as not correct. Split: {split}."]
    if missing:
        d1_notes.append(f"No usable D1 results for {missing}.")
    excluded = {s: int((d1[s]["status"] == "api_error").sum()) for s in d1}
    if any(excluded.values()):
        d1_notes.append(f"Systems are NOT comparable here: API errors removed problems unevenly {excluded}. "
                        "Rerun to fill them in.")
    tables.append(("T2", "Accuracy by topic x system", t2_t3_accuracy(graded, "topic", TOPICS, base), d1_notes))
    tables.append(("T3", "Accuracy by size x system", t2_t3_accuracy(graded, "size", SIZES, base), d1_notes))
    tables.append(("T2b", "RQ1 wrong-claim rate", t2b_wrong_claim(graded, base, bootstrap), d1_notes[1:] + [
        "Share of responses with at least 1 delivered claim that contradicts the ground truth, graded by "
        "grader.py and never by the verifier. Mean, SD and range are over runs; the 95% CI comes from a "
        f"bootstrap of {bootstrap} resamples clustered by problem.",
        "Only delivered claims count: OURS withholds refuted claims. A response that abstains or gives no usable "
        "reply delivers no claims, so it cannot be wrong. Read this with the abstention and answer-withheld rates "
        "in T3b.",
        "The human audit of 10% of claim extractions (PREREGISTRATION.md s3) has not been done.",
    ]))
    tables.append(("T3d", "RQ4 certified-claim coverage (OURS)", t3d_coverage(graded.get(S0), bootstrap), [
        "Certified claims / all extracted claims in the final version of each response, withheld claims "
        "included, as registered. A row operation "
        "is certified only if applying its stated operation reproduces the next matrix exactly. Claims that "
        "pass on row-equivalence alone are weakly checked: they never count as certified, and are reported "
        "separately. Decidable share is (certified + refuted) / all.",
        "Blank if OURS has not been run. Coverage of student steps (prereg-v1 T3) needs D3.",
    ]))
    tables.append(("T3b", "Wrong answers, fabricated answers and abstention", t3b_reliability(graded, base, d1), [
        "Consistent computation error: wrong, but the response's own working arrives at it. Over all problems.",
        "Fabricated: wrong, not derived from the working (none shown, or it differs from the last value the "
        "working computes), and passes the surface check. Over eigenpairs and diagonalisation only: eigenvalues "
        "must sum to trace(A), or trace(D) must equal trace(A).",
    ]))
    tables.append(("T3c", "Fabricated rate by topic", t3c_fabricated(graded, base), [
        "n/a: the topic has no surface check.",
    ]))
    frame, notes = t4_student(d3, d3_frame)
    if d3_frame is None:
        notes = [f"Blank: {d3_note}. T4 needs real student solutions, which must not be invented."]
    elif not d3:
        notes.append("Blank: D3 data exists but no system has been run on it.")
    tables.append(("T4", "First wrong step, step F1, misconception macro-F1 (D3)", frame, notes))
    frame, notes = t4b_kappa(d3_frame)
    if d3_frame is None:
        notes = [f"Blank: {d3_note}."]
    tables.append(("T4b", "Cohen's kappa between the two D3 annotators", frame, notes))
    tables.append(("T5", "Hint leak rate and hint-verified rate", t5_hints(graded, base), [
        "D1 hints are exploratory and counted in run 1 only. "
        "Both rates are over delivered hints, measured by the same checks for every system. Leaks are "
        "checked against the stored ground truth. B1, B2 and B3 are measured but never filtered. A hint "
        "with no computed values counts as verified, so the last column shows how many passed with at least "
        "one certified claim, i.e. checked mathematics rather than a vacuous pass.",
    ]))
    rows = t6_ablations(graded, ["OURS"] + ABLATIONS)
    t6_notes = ["A blank row means that configuration has not been run on this split.",
                "A1 is identical to B4 and A3 to OURS on D1 by construction: the Diagnostic Agent needs student "
                "work, which only D3 has."]
    tables.append(("T6", "Ablations (D1)", rows, t6_notes))
    tables.append(("T7", "Mean cost per problem", t7_cost(graded, base), [
        "Calls, tokens and latency are those of the original API call, so cached reruns report the same cost. "
        "Seconds are the latency of the successful API call plus local compute. Pacing and rate-limit waits are excluded.",
        "Seconds come only from Groq synchronous runs; batch runs record no latency.",
        "Output tokens are all tokens except the prompt, so they include thinking.",
    ]))
    frame, notes = t7b_truncation(graded, base)
    tables.append(("T7b", "Truncation rate by system and size", frame, notes))
    rq2 = rq2_frames(d3, d3_data.gold_labels(d3_frame)[0]) if d3_frame is not None and d3 else {}
    frame, notes = t8_tests(graded, rq2)
    tables.append(("T8", "McNemar tests with Holm correction (primary family)", frame, notes))
    return tables


def write(tables: list, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, title, frame, notes in tables:
        frame.to_csv(out / f"{key}.csv", index=False)
        lines += [f"## {key}. {title}", "", markdown(frame), ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build tables T1 to T7 from result files.")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    tags = model_tags(args.split) or [None]
    for tag in tags:
        out = RESULTS / "tables" / args.split / (tag or "no_results")
        write(build(args.split, tag), out)
        print(f"# {tag or 'no results'}\n")
        print((out / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
