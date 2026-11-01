"""T1 (D1 part): dataset statistics of the certified problem bank. No LLM involved.
python scripts/d1_stats.py  ->  results/T1_d1_bank.csv"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latutor.exact import num  # noqa: E402
from latutor.generator import load  # noqa: E402

rows = []
probs = load(ROOT / "data/d1/problems.jsonl")
for topic in ["rref", "det", "inverse", "eigen", "diag", "lsq"]:
    for n in [3, 4, 5]:
        ps = [p for p in probs if p["topic"] == topic and p["size"] == n]
        r = {"topic": topic, "size": n, "problems": len(ps),
             "dev": sum(p["split"] == "dev" for p in ps), "test": sum(p["split"] == "test" for p in ps),
             "A_shape": f"{len(ps[0]['A'])}x{len(ps[0]['A'][0])}",
             "mean_abs_entry": round(sum(abs(int(x)) for p in ps for row in p["A"] for x in row)
                                     / sum(len(p["A"]) * len(p["A"][0]) for p in ps), 2)}
        s = [p["solution"] for p in ps]
        if topic == "rref":
            r["rank_min_max"] = f"{min(x['rank'] for x in s)}-{max(x['rank'] for x in s)}"
            r["rank_deficient"] = sum(x["rank"] < n for x in s)
        if topic == "det":
            d = [abs(num(x["determinant"])) for x in s]
            r["abs_det_median"] = sorted(d)[len(d) // 2]
        if topic == "inverse":
            r["abs_det_range"] = f"{min(abs(num(x['determinant'])) for x in s)}-{max(abs(num(x['determinant'])) for x in s)}"
            r["max_denominator"] = max(num(e).q for x in s for row in x["inverse"] for e in row)
        if topic in ("eigen", "diag"):
            r["with_repeated_eigenvalue"] = sum(any(e["algebraic_multiplicity"] > 1 for e in x["eigenvalues"]) for x in s)
            r["eigenvalue_range"] = f"{min(num(e['value']) for x in s for e in x['eigenvalues'])} to {max(num(e['value']) for x in s for e in x['eigenvalues'])}"
        if topic == "lsq":
            r["max_denominator"] = max(num(e).q for x in s for e in x["x_hat"])
        rows.append(r)
keys = list(dict.fromkeys(k for r in rows for k in r))
out = ROOT / "results/T1_d1_bank.csv"
out.parent.mkdir(exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
