"""Score result files into tables T1 to T7, as CSV and markdown.

  python score.py --split dev

Writes results/tables/<split>/T*.csv and tables.md. A cell that cannot be
computed stays blank, and the note under the table says why. No number here
comes from anywhere but a result row or a dataset file.
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
from run import RESULTS
from systems import ABLATIONS, SYSTEMS
from verifier import leak_check, verify_hint

TOPICS = gen_d1.TOPICS
SIZES = gen_d1.SIZES


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
    """results/<system>_<dataset>_<split>_run<k>.csv -> (system, dataset)."""
    parts = path.stem.split("_")
    if len(parts) != 4 or parts[2] != split or parts[1] not in ("d1", "d2", "d3") or not parts[3].startswith("run"):
        return None
    return parts[0], parts[1]


def load_results(split: str, results: Path = RESULTS) -> dict:
    """Runs of the same system and dataset are pooled, so counts are over problem x run."""
    frames: dict = {}
    for path in sorted(results.glob(f"*_{split}_run*.csv")):
        parsed = parse_name(path, split)
        if parsed:
            frame = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"pred_misconception": str})
            frames.setdefault(parsed, []).append(frame)
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}


def _delivered(row) -> list:
    return [h for h in json.loads(row["hints_json"]) if not h.get("withheld") and h.get("text")]


def grade_d1(frame: pd.DataFrame, problems: dict) -> pd.DataFrame:
    """Add grade columns. api_error rows are dropped: they say nothing about a system."""
    usable = frame[frame["status"] != "api_error"].copy()
    correct, wrong, consistent, fabricated, delivered, leaks, verified, with_claim, n_hints = ([] for _ in range(9))
    for _, row in usable.iterrows():
        p = problems[row["problem_id"]]
        answer = json.loads(row["answer_json"])
        has_answer = row["status"] == "answered" and answer is not None
        ok = has_answer and grader.grade(p, answer)
        trace = json.loads(row["trace_json"])
        correct.append(ok)
        wrong.append(has_answer and not ok)
        delivered.append(has_answer)
        consistent.append(has_answer and grader.consistent_computation_error(p, answer, trace))
        fabricated.append(grader.fabricated(p, answer if has_answer else None, trace))
        hints = _delivered(row)
        truth = grader.answer_of(p)
        leaks.append(sum(leak_check(h["text"], p["topic"], truth, h.get("claims")).leaked for h in hints))
        checks = [verify_hint(p, h["text"], h.get("claims")) for h in hints]
        verified.append(sum(c.verified for c in checks))
        with_claim.append(sum(c.verified and c.n_certified > 0 for c in checks))
        n_hints.append(len(hints))
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
        rows.append({
            "system": name,
            "problems": len(f),
            "mean_llm_calls": _num(f["llm_calls"].mean(), 2),
            "mean_tokens": _num(f["total_tokens"].mean(), 0),
            "mean_seconds": _num((f["llm_seconds"] + f["local_seconds"]).mean(), 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output
def markdown(frame: pd.DataFrame) -> str:
    def clean(x):
        return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def build(split: str, results: Path = RESULTS) -> list:
    problems = gen_d1.load()
    by_id = {p["id"]: p for p in problems}
    loaded = load_results(split, results)
    d1 = {name: f for (name, ds), f in loaded.items() if ds == "d1"}
    d3 = {name: f for (name, ds), f in loaded.items() if ds == "d3"}
    graded = {name: grade_d1(f, by_id) for name, f in d1.items()}
    base = [s for s in SYSTEMS if s in graded]

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
        d1_notes.append(f"No D1 results for {missing}.")
    excluded = {s: int((d1[s]["status"] == "api_error").sum()) for s in base}
    if any(excluded.values()):
        d1_notes.append(f"Systems are NOT comparable here: API errors removed problems unevenly {excluded}. "
                        "Rerun to fill them in.")
    tables.append(("T2", "Accuracy by topic x system", t2_t3_accuracy(graded, "topic", TOPICS, base), d1_notes))
    tables.append(("T3", "Accuracy by size x system", t2_t3_accuracy(graded, "size", SIZES, base), d1_notes))
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
    ]))
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
    out = RESULTS / "tables" / args.split
    write(build(args.split), out)
    print((out / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
