"""Engineering validation of the Verification Agent on D1 (no LLM involved).

For every D1 problem (dev and test; no tutor is run, so the test split is not
touched by any system) this script builds:
  * valid claims  - the stored answer, plus equivalent valid forms that differ from it
                    (rescaled eigenvectors, rescaled P columns, the other characteristic-polynomial
                    sign convention, random row operations of A, cofactor expansions, row swaps);
  * invalid claims - single, typical mistakes (off-by-one entry, sign error, dropped eigenvalue,
                    wrong multiplicity, non-eigenvector, mismatched D, arithmetic slip in a row step).

It reports, by topic and size:
  * false rejections - valid claims the verifier did not certify,
  * false acceptances - invalid claims the verifier certified,
  * verifier/grader disagreements - the verifier (certificate from A) and the grader
    (stored ground truth) must agree on every claim.

This is a software check, not one of the pre-registered research questions.
Usage:  python scripts/validate_verifier.py [--problems data/d1/problems.jsonl] [--out results/verifier_validation]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latutor.exact import mat, num, to_json, vec  # noqa: E402
from latutor.generator import load  # noqa: E402
from latutor.grader import Grader, gold_final_claims  # noqa: E402
from latutor.verifier import Verifier  # noqa: E402

SEED = 20261024


def _bump(M, rng, delta=1):
    M = M.copy()
    i, j = rng.randrange(M.rows), rng.randrange(M.cols)
    M[i, j] += delta
    return M, (i, j)


def _row_ops(M, rng, k=3):
    """k random invertible row operations (valid intermediate step)."""
    M = M.copy()
    for _ in range(k):
        op = rng.choice(["add", "swap", "scale"])
        i, j = rng.sample(range(M.rows), 2)
        if op == "add":
            M[i, :] = M[i, :] + rng.choice([-3, -2, -1, 1, 2, 3]) * M[j, :]
        elif op == "swap":
            M.row_swap(i, j)
        else:
            M[i, :] = rng.choice([-2, 2, 3, sp.Rational(1, 2)]) * M[i, :]
    return M


def _arith_slip(M, rng):
    """One row operation with an arithmetic slip in a single entry (invalid step)."""
    good = _row_ops(M, rng, k=1)
    for _ in range(20):
        bad, _ = _bump(good, rng, rng.choice([-1, 1]))
        if bad.col_join(M).rank() != M.rank() or bad.rank() != M.rank():
            return bad
    return None  # could not make a detectable slip (e.g. full-rank square with slip staying full rank)


def cases_for(p, rng):
    """Yield (label, claim, expected_valid)."""
    A = mat(p["A"])
    n = A.rows
    s, t = p["solution"], p["topic"]
    for c in gold_final_claims(p):
        yield ("gold:" + c["type"], c, True)

    if t == "rref":
        R = mat(s["rref"])
        bad, _ = _bump(R, rng)
        yield ("bad:rref_entry", {"type": "rref", "matrix": to_json(bad)}, False)
        yield ("bad:rank_off_by_one", {"type": "rank", "value": s["rank"] + rng.choice([-1, 1])}, False)
        yield ("ok:row_ops_of_A", {"type": "intermediate_matrix", "context": "A",
                                   "matrix": to_json(_row_ops(A, rng))}, True)
        slip = _arith_slip(A, rng)
        if slip is not None:
            yield ("bad:row_op_slip", {"type": "intermediate_matrix", "context": "A",
                                       "matrix": to_json(slip)}, False)

    if t == "det":
        d = num(s["determinant"])
        yield ("bad:det_off_by_one", {"type": "determinant", "value": to_json(d + 1)}, False)
        if d != 0:
            yield ("bad:det_sign", {"type": "determinant", "value": to_json(-d)}, False)
        i, j = rng.sample(range(n), 2)
        S = A.copy(); S.row_swap(i, j)
        yield ("ok:det_row_swap", {"type": "det_relation", "coefficient": -1, "matrix": to_json(S)}, True)
        if d != 0:
            yield ("bad:det_row_swap_no_sign", {"type": "det_relation", "coefficient": 1,
                                                "matrix": to_json(S)}, False)
        terms = [{"coefficient": to_json((-1) ** j * A[0, j]), "matrix": to_json(A.minor_submatrix(0, j))}
                 for j in range(n)]
        yield ("ok:cofactor_row1", {"type": "det_combination", "terms": terms}, True)
        bad_terms = [dict(tm) for tm in terms]
        k = next((j for j in range(n) if A[0, j] != 0 and A.minor_submatrix(0, j).det() != 0), None)
        if k is not None:
            bad_terms[k] = {"coefficient": to_json(-((-1) ** k) * A[0, k]), "matrix": terms[k]["matrix"]}
            yield ("bad:cofactor_sign", {"type": "det_combination", "terms": bad_terms}, False)

    if t == "inverse":
        B = mat(s["inverse"])
        bad, _ = _bump(B, rng)
        yield ("bad:inverse_entry", {"type": "inverse", "matrix": to_json(bad)}, False)
        if B.T != B:
            yield ("bad:inverse_transposed", {"type": "inverse", "matrix": to_json(B.T)}, False)
        aug = A.row_join(sp.eye(n))
        yield ("ok:augmented_row_ops", {"type": "intermediate_matrix", "context": "augmented_A_I",
                                        "matrix": to_json(_row_ops(aug, rng))}, True)
        slip = _arith_slip(aug, rng)
        if slip is not None:
            yield ("bad:augmented_slip", {"type": "intermediate_matrix", "context": "augmented_A_I",
                                          "matrix": to_json(slip)}, False)

    if t in ("eigen", "diag"):
        eig = s["eigenvalues"]
        lams = [num(e["value"]) for e in eig]
        cand = next(x for x in range(-20, 21) if sp.Integer(x) not in lams)
        yield ("bad:not_an_eigenvalue", {"type": "eigenvalue", "value": cand}, False)
        if len(eig) > 1:
            yield ("bad:dropped_eigenvalue", {"type": "eigenvalues", "values": eig[1:]}, False)
        e0 = eig[0]
        yield ("bad:wrong_multiplicity", {"type": "eigenvalue", "value": e0["value"],
                                          "algebraic_multiplicity": e0["algebraic_multiplicity"] + 1}, False)
        sp0 = s["eigenspaces"][0]
        v = vec(sp0["basis"][0])
        yield ("ok:scaled_eigenvector", {"type": "eigenvector", "eigenvalue": sp0["eigenvalue"],
                                         "vector": to_json(-2 * v)}, True)
        # a vector outside the eigenspace
        for k in range(n):
            w = v + sp.eye(n).col(k)
            if (A - num(sp0["eigenvalue"]) * sp.eye(n)) * w != sp.zeros(n, 1):
                yield ("bad:non_eigenvector", {"type": "eigenvector", "eigenvalue": sp0["eigenvalue"],
                                               "vector": to_json(w)}, False)
                break
        cp = [num(x) for x in s["characteristic_polynomial"]]
        yield ("ok:charpoly_other_sign", {"type": "characteristic_polynomial",
                                          "coefficients": to_json([(-1) ** n * x for x in cp])}, True)
        cpb = list(cp); cpb[-1] += 1
        yield ("bad:charpoly_constant", {"type": "characteristic_polynomial", "coefficients": to_json(cpb)}, False)
        lam = lams[0]
        yield ("ok:A_minus_lambda_I_row_ops", {"type": "intermediate_matrix", "context": "A_minus_lambda_I",
                                               "lambda": to_json(lam),
                                               "matrix": to_json(_row_ops(A - lam * sp.eye(n), rng))}, True)
        if t == "diag":
            P, D = mat(s["P"]), mat(s["D"])
            Ps = P * sp.diag(*[rng.choice([-1, 2, 3]) for _ in range(n)])
            yield ("ok:rescaled_P", {"type": "diagonalisation", "P": to_json(Ps), "D": to_json(D)}, True)
            d = [D[i, i] for i in range(n)]
            pair = next(((i, j) for i in range(n) for j in range(i + 1, n) if d[i] != d[j]), None)
            if pair:
                i, j = pair
                d2 = list(d); d2[i], d2[j] = d2[j], d2[i]
                yield ("bad:D_order_mismatch", {"type": "diagonalisation", "P": to_json(P),
                                                "D": to_json(sp.diag(*d2))}, False)

    if t == "lsq":
        x = vec(s["x_hat"])
        bad = x.copy(); bad[rng.randrange(x.rows)] += sp.Rational(1, 7)
        yield ("bad:lsq_entry", {"type": "lsq_solution", "vector": to_json(bad)}, False)
        AtA = mat(s["AtA"])
        bad2, _ = _bump(AtA, rng)
        yield ("bad:AtA_entry", {"type": "normal_equations", "AtA": to_json(bad2)}, False)
        aug = AtA.row_join(vec(s["Atb"]))
        yield ("ok:normal_aug_row_ops", {"type": "intermediate_matrix", "context": "normal_augmented",
                                         "matrix": to_json(_row_ops(aug, rng))}, True)
        # solving Ax = b on the square part ignores the least-squares structure: a classic mistake
        b = vec(p["b"])
        k = x.rows
        A_sq = mat(p["A"])[:k, :]
        if A_sq.det() != 0:
            naive = A_sq.LUsolve(b[:k, :])
            if naive != x:
                yield ("bad:solve_square_subsystem", {"type": "lsq_solution", "vector": to_json(naive)}, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/d1/problems.jsonl")
    ap.add_argument("--out", default="results/verifier_validation")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows, t0 = [], time.time()
    for p in load(args.problems):
        ver, gr = Verifier(p), Grader(p)
        for label, claim, valid in cases_for(p, rng):
            t1 = time.perf_counter()
            v = ver.check(claim)
            ms = (time.perf_counter() - t1) * 1000
            g = gr.grade(claim)
            rows.append({"problem_id": p["id"], "topic": p["topic"], "size": p["size"], "split": p["split"],
                         "case": label, "expected_valid": valid, "verifier": v.status,
                         "verifier_reason": v.reason, "grader": g, "verify_ms": round(ms, 2)})
    with open(out / "cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for key in [(r["topic"], r["size"]), (r["topic"], "all"), ("all", "all")]:
            a = agg[key]
            if r["expected_valid"]:
                a["valid"] += 1
                a["false_reject"] += r["verifier"] != "certified"
            else:
                a["invalid"] += 1
                a["false_accept"] += r["verifier"] == "certified"
            a["uncheckable"] += r["verifier"] == "uncheckable"
            vg = {"certified": "correct", "refuted": "wrong"}.get(r["verifier"])
            a["disagree"] += vg is not None and vg != r["grader"]
            a["grader_mismatch"] += (r["grader"] == "correct") != r["expected_valid"]
            a["ms_total"] += r["verify_ms"]
            a["n"] += 1
    summary = []
    for (topic, size), a in sorted(agg.items(), key=lambda kv: (kv[0][0] == "all", kv[0][0], str(kv[0][1]))):
        summary.append({"topic": topic, "size": size, "claims": a["n"], "valid": a["valid"],
                        "invalid": a["invalid"], "false_rejections": a["false_reject"],
                        "false_acceptances": a["false_accept"], "uncheckable": a["uncheckable"],
                        "verifier_grader_disagreements": a["disagree"],
                        "grader_vs_expected_mismatches": a["grader_mismatch"],
                        "mean_verify_ms": round(a["ms_total"] / a["n"], 2)})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    meta = {"problems": args.problems, "seed": SEED, "cases": len(rows),
            "wall_seconds": round(time.time() - t0, 1), "sympy": sp.__version__,
            "python": sys.version.split()[0]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for s_ in summary:
        if s_["size"] == "all":
            print(s_)
    print(meta)


if __name__ == "__main__":
    main()
