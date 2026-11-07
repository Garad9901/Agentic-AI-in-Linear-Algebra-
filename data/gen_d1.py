"""Generate dataset D1: python data/gen_d1.py [--n 15] [--seed 20261024]

6 topics x sizes 3, 4, 5 x N problems per cell (270 at the default, prereg-v1 s4.1),
integer entries in [-9, 9],
exact ground truth from SymPy, split 20% dev / 80% test within each
(topic, size) cell. Each cell has its own RNG, so a larger --n extends a smaller
one without changing the problems they share.

"Size n" means A is n x n, except least squares, where A is (n+1) x n with full
column rank and b is not in col(A).

The construction limits below come from the approved deviations in
la_repo/docs/DEVIATIONS.md. All of them stay inside the [-9, 9] requirement:
  rref_rank:  rank r drawn from {n-2, n-1, n}, so rank is not always full; A = B C for
              r < n and a random full-rank A for r = n
  inverse:    |det A| <= 100, so answers stay readable fractions
  least_sq:   entries of A in [-4, 4], det(A^T A) <= 600
  eigen:      P unimodular, A rejected if >40% zeros or reducible, so eigenvalues
              cannot be read off a block; repeated eigenvalue with probability 0.3
              (eigenpairs) or 0.5 (diagonalisation)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.exact import to_json  # noqa: E402

SEED = 20261024
NAMESPACE = "prereg-v1"  # keeps this bank disjoint from the bank used for prompt dry runs (docs/DEVIATIONS.md)
TOPICS = ["rref_rank", "determinant", "inverse", "eigenpairs", "diagonalisation", "least_squares"]
SIZES = [3, 4, 5]
DEFAULT_N = 15
DEV_SHARE = 0.2
ENTRY_MAX = 9
MAX_TRIES_PER_PROBLEM = 2000

QUESTIONS = {
    "rref_rank": "Find the reduced row echelon form (RREF) of A and the rank of A.",
    "determinant": "Find the determinant of A.",
    "inverse": "Find the inverse of A.",
    "eigenpairs": "Find all eigenvalues of A and an eigenvector for each. List one eigenpair per eigenvalue "
                  "counted with algebraic multiplicity, using independent eigenvectors for a repeated eigenvalue.",
    "diagonalisation": "Diagonalise A: find an invertible matrix P and a diagonal matrix D with A = P D P^(-1).",
    "least_squares": "Find the least-squares solution x that minimises ||A x - b||.",
}


def _rand_mat(rng, r, c, lo, hi):
    return sp.Matrix(r, c, lambda i, j: rng.randint(lo, hi))


def _within(M, bound=ENTRY_MAX):
    return max(abs(e) for e in M) <= bound


def _unimodular(rng, n):
    """Random dense integer matrix with det = +-1: row-permuted L U with unit diagonals."""
    L, U = sp.eye(n), sp.eye(n)
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = rng.choice([-1, 0, 0, 1])
            elif i < j:
                U[i, j] = rng.choice([-1, 0, 0, 1])
    perm = list(range(n))
    rng.shuffle(perm)
    return (L * U).extract(perm, list(range(n)))


def _irreducible(M):
    """True if A is not permutation-similar to block-triangular form."""
    n = M.rows

    def reach(adj):
        seen, stack = {0}, [0]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j not in seen and adj(i, j):
                    seen.add(j)
                    stack.append(j)
        return len(seen) == n

    return reach(lambda i, j: i != j and M[i, j] != 0) and reach(lambda i, j: i != j and M[j, i] != 0)


def _zero_frac(M):
    return sum(1 for e in M if e == 0) / (M.rows * M.cols)


def _gen_rref_rank(rng, n):
    # r is drawn once: redrawing it after a rejection would almost never leave full rank at n >= 4,
    # because a full-rank product B C rarely has every entry within [-9, 9].
    r = rng.randint(max(1, n - 2), n)
    while True:
        if r == n:
            A = _rand_mat(rng, n, n, -ENTRY_MAX, ENTRY_MAX)
        else:
            A = _rand_mat(rng, n, r, -3, 3) * _rand_mat(rng, r, n, -3, 3)
        if A.rank() == r and _within(A) and all(any(A.row(i)) for i in range(n)):
            R, piv = A.rref()
            return {"A": A}, {"rref": R, "rank": r, "pivots": [p + 1 for p in piv]}


def _gen_determinant(rng, n):
    A = _rand_mat(rng, n, n, -ENTRY_MAX, ENTRY_MAX)
    return {"A": A}, {"determinant": A.det(method="bareiss")}


def _gen_inverse(rng, n):
    while True:
        A = _rand_mat(rng, n, n, -ENTRY_MAX, ENTRY_MAX)
        d = A.det(method="bareiss")
        if d != 0 and abs(d) <= 100:
            return {"A": A}, {"inverse": A.inv()}


def _eigen_truth(A):
    values = [{"value": lam, "multiplicity": int(m)} for lam, m in sorted(A.eigenvals().items(), key=lambda t: t[0])]
    spaces = [{"eigenvalue": lam, "basis": [sp.Matrix(v) for v in basis]}
              for lam, _, basis in sorted(A.eigenvects(), key=lambda t: t[0])]
    return {"eigenvalues": values, "eigenspaces": spaces}


def _gen_pdp(rng, n, repeat):
    while True:
        P = _unimodular(rng, n)
        if repeat:
            vals = [rng.randint(-4, 4) for _ in range(n - 1)]
            vals.append(rng.choice(vals))
        else:
            vals = rng.sample(range(-5, 6), n)
        rng.shuffle(vals)
        D = sp.diag(*vals)
        A = P * D * P.inv()
        if _within(A) and _zero_frac(A) <= 0.4 and _irreducible(A) and all(e.is_integer for e in A):
            return A, P, D


def _gen_eigenpairs(rng, n):
    A, _, _ = _gen_pdp(rng, n, repeat=rng.random() < 0.3)
    return {"A": A}, _eigen_truth(A)


def _gen_diagonalisation(rng, n):
    A, P, D = _gen_pdp(rng, n, repeat=rng.random() < 0.5)
    return {"A": A}, {"P": P, "D": D, **_eigen_truth(A)}


def _gen_least_squares(rng, n):
    m = n + 1
    while True:
        A = _rand_mat(rng, m, n, -4, 4)
        AtA = A.T * A
        d = AtA.det(method="bareiss")
        if d == 0 or d > 600:
            continue
        b = _rand_mat(rng, m, 1, -ENTRY_MAX, ENTRY_MAX)
        x = AtA.LUsolve(A.T * b)
        if A * x == b:
            continue  # b in col(A): not a genuine least-squares problem
        return {"A": A, "b": b}, {"x": x}


_GEN = {"rref_rank": _gen_rref_rank, "determinant": _gen_determinant, "inverse": _gen_inverse,
        "eigenpairs": _gen_eigenpairs, "diagonalisation": _gen_diagonalisation,
        "least_squares": _gen_least_squares}


def _problem_text(topic, data):
    text = f"{QUESTIONS[topic]}\nA = {to_json(data['A'])}"
    if "b" in data:
        text += f"\nb = {to_json(data['b'])}"
    return text


def generate(n_per_cell: int = DEFAULT_N, seed: int = SEED) -> list[dict]:
    problems = []
    for topic in TOPICS:
        for size in SIZES:
            rng = random.Random(f"{seed}-{NAMESPACE}-{topic}-{size}")
            cell, seen = [], set()
            tries = 0
            while len(cell) < n_per_cell:
                tries += 1
                if tries > MAX_TRIES_PER_PROBLEM * n_per_cell:
                    raise RuntimeError(f"cannot find {n_per_cell} distinct problems for {topic} size {size}")
                data, truth = _GEN[topic](rng, size)
                key = json.dumps([to_json(data["A"]), to_json(data.get("b"))])
                if key in seen:
                    continue
                seen.add(key)
                rec = {
                    "id": f"D1-{topic}-{size}-{len(cell) + 1:03d}",
                    "dataset": "d1",
                    "topic": topic,
                    "size": size,
                    "problem": _problem_text(topic, data),
                    "A": to_json(data["A"]),
                    "ground_truth": to_json(truth),
                }
                if "b" in data:
                    rec["b"] = to_json(data["b"])
                cell.append(rec)
            order = list(range(n_per_cell))
            random.Random(f"{seed}-{NAMESPACE}-split-{topic}-{size}").shuffle(order)
            dev = set(order[:round(DEV_SHARE * n_per_cell)])
            for k, rec in enumerate(cell):
                rec["split"] = "dev" if k in dev else "test"
            problems.extend(cell)
    return problems


def write(path: Path, problems: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for p in problems:
            f.write(json.dumps(p, separators=(",", ":")) + "\n")


def load(path: Path = ROOT / "data" / "d1.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="problems per (topic, size)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "d1.jsonl")
    args = ap.parse_args()
    problems = generate(args.n, args.seed)
    write(args.out, problems)
    dev = sum(p["split"] == "dev" for p in problems)
    print(f"wrote {len(problems)} problems to {args.out} ({dev} dev, {len(problems) - dev} test)")


if __name__ == "__main__":
    main()
