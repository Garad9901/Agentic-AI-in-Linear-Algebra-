"""Certified problem bank D1 (PREREGISTRATION.md section 4.1).

Fixed by the plan:
  seed 20261024; 6 topics x 3 sizes (3, 4, 5) x 15 problems = 270;
  integer entries in [-9, 9]; eigen and diagonalisation problems built as
  P D P^-1 with unimodular P (integer eigenvalues, exact certificates);
  split dev 20% / test 80%, stratified by topic and size.

Generator choices the plan left open (listed in docs/PROPOSED_DEVIATIONS.md and to be
logged in docs/DEVIATIONS.md as clarifications before the first test run):
  * least squares "size n" means A is (n+1) x n with full column rank and b not in col(A),
    so the normal-equation matrix A^T A is n x n;
  * rank problems use rank r drawn from {n-2, n-1, n} (min 1), built as B C;
  * inverse problems require |det A| <= 100 so answers stay readable fractions;
  * eigen/diag problems reject A with more than 40% zero entries and reducible A
    (permutation-similar to block-triangular, so an eigenvalue could be read off);
    P is a row-permuted unit L U with entries in {-1, 0, 1};
  * diagonalisation problems include a repeated eigenvalue with probability 1/2.

The stored solution is the ground truth used by grader.py. It is computed here,
before any tutor runs (circularity guard, PREREGISTRATION.md section 3).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sympy as sp

from .exact import to_json

SEED = 20261024
TOPICS = ["rref", "det", "inverse", "eigen", "diag", "lsq"]
SIZES = [3, 4, 5]
PER_CELL = 15
DEV_PER_CELL = 3  # 20% of 15
ENTRY_MAX = 9

QUESTIONS = {
    "rref": "Find the reduced row echelon form (RREF) of A, its rank, and its pivot columns.",
    "det": "Find the determinant of A.",
    "inverse": "Find the inverse of A.",
    "eigen": "Find all eigenvalues of A with their algebraic multiplicities, and a basis of eigenvectors for each eigenvalue.",
    "diag": "Diagonalise A: find an invertible matrix P and a diagonal matrix D such that A = P D P^(-1).",
    "lsq": "Find the least-squares solution x_hat that minimises ||A x - b||.",
}


def _rand_mat(rng, r, c, lo, hi):
    return sp.Matrix(r, c, lambda i, j: rng.randint(lo, hi))


def _within(M, bound=ENTRY_MAX):
    return max(abs(e) for e in M) <= bound


def _is_triangular(M):
    return M.is_upper or M.is_lower


def _unimodular(rng, n):
    """Random dense integer matrix with det = +-1: row-permuted L U, unit triangular, entries in {-1,0,1}."""
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
    """True if the directed graph of non-zero off-diagonal entries is strongly connected,
    i.e. A is not permutation-similar to a block-triangular matrix (no eigenvalue readable
    from an isolated block)."""
    n = M.rows
    def reach(adj):
        seen, stack = {0}, [0]
        while stack:
            i = stack.pop()
            for j in range(n):
                if j not in seen and adj(i, j):
                    seen.add(j); stack.append(j)
        return len(seen) == n
    return reach(lambda i, j: i != j and M[i, j] != 0) and reach(lambda i, j: i != j and M[j, i] != 0)


def _zero_frac(M):
    return sum(1 for e in M if e == 0) / (M.rows * M.cols)


def _gen_rref(rng, n):
    while True:
        r = rng.randint(max(1, n - 2), n)
        B = _rand_mat(rng, n, r, -3, 3)
        C = _rand_mat(rng, r, n, -3, 3)
        A = B * C
        if A.rank() == r and _within(A) and all(any(A.row(i)) for i in range(n)):
            R, piv = A.rref()
            return {"A": A}, {"rref": R, "rank": r, "pivot_columns": [p + 1 for p in piv]}


def _gen_det(rng, n):
    A = _rand_mat(rng, n, n, -ENTRY_MAX, ENTRY_MAX)
    return {"A": A}, {"determinant": A.det(method="bareiss")}


def _gen_inverse(rng, n):
    while True:
        A = _rand_mat(rng, n, n, -ENTRY_MAX, ENTRY_MAX)
        d = A.det(method="bareiss")
        if d != 0 and abs(d) <= 100:
            return {"A": A}, {"inverse": A.inv(), "determinant": d}


def _eigen_solution(A):
    lam_mult = A.eigenvals()  # exact; integer by construction
    spaces = {}
    for lam, _mult, basis in A.eigenvects():
        spaces[lam] = [sp.Matrix(v) for v in basis]
    charpoly = A.charpoly(sp.Symbol("lam")).all_coeffs()  # det(lam I - A), highest degree first
    return lam_mult, spaces, charpoly


def _gen_pdp(rng, n, force_repeat):
    while True:
        P = _unimodular(rng, n)
        if force_repeat:
            vals = [rng.randint(-4, 4) for _ in range(n - 1)]
            vals.append(rng.choice(vals))
        else:
            vals = rng.sample(range(-5, 6), n)
        rng.shuffle(vals)
        D = sp.diag(*vals)
        A = P * D * P.inv()
        if (_within(A) and _zero_frac(A) <= 0.4 and _irreducible(A)
                and all(e.is_integer for e in A)):
            return A, P, D


def _gen_eigen(rng, n):
    A, _P, _D = _gen_pdp(rng, n, force_repeat=rng.random() < 0.3)
    lam_mult, spaces, charpoly = _eigen_solution(A)
    return {"A": A}, {
        "diagonalisable": True,
        "eigenvalues": {lam: m for lam, m in lam_mult.items()},
        "eigenspaces": spaces,
        "characteristic_polynomial": charpoly,
    }


def _gen_diag(rng, n):
    A, P, D = _gen_pdp(rng, n, force_repeat=rng.random() < 0.5)
    lam_mult, spaces, charpoly = _eigen_solution(A)
    return {"A": A}, {
        "diagonalisable": True,
        "P": P,
        "D": D,
        "eigenvalues": {lam: m for lam, m in lam_mult.items()},
        "eigenspaces": spaces,
        "characteristic_polynomial": charpoly,
    }


def _gen_lsq(rng, n):
    m = n + 1
    while True:
        A = _rand_mat(rng, m, n, -4, 4)
        if A.rank() != n:
            continue
        AtA = A.T * A
        if AtA.det() > 600:
            continue
        b = _rand_mat(rng, m, 1, -ENTRY_MAX, ENTRY_MAX)
        Atb = A.T * b
        x = AtA.LUsolve(Atb)
        r = A * x - b
        if r == sp.zeros(m, 1):
            continue  # b in col(A): not a genuine least-squares problem
        return {"A": A, "b": b}, {"x_hat": x, "AtA": AtA, "Atb": Atb, "residual": r}


_GEN = {"rref": _gen_rref, "det": _gen_det, "inverse": _gen_inverse,
        "eigen": _gen_eigen, "diag": _gen_diag, "lsq": _gen_lsq}


def _solution_to_json(topic, sol):
    out = {}
    for k, v in sol.items():
        if k == "eigenvalues":
            out[k] = [{"value": to_json(lam), "algebraic_multiplicity": int(m)}
                      for lam, m in sorted(v.items(), key=lambda t: t[0])]
        elif k == "eigenspaces":
            out[k] = [{"eigenvalue": to_json(lam), "basis": [to_json(b) for b in basis]}
                      for lam, basis in sorted(v.items(), key=lambda t: t[0])]
        else:
            out[k] = to_json(v)
    return out


def generate(seed: int = SEED):
    """Return the list of 270 problem records, in a fixed order, with dev/test split."""
    rng = random.Random(seed)
    problems = []
    for topic in TOPICS:
        for n in SIZES:
            cell = []
            for k in range(PER_CELL):
                data, sol = _GEN[topic](rng, n)
                rec = {
                    "id": f"D1-{topic}-{n}-{k + 1:02d}",
                    "dataset": "D1",
                    "topic": topic,
                    "size": n,
                    "question": QUESTIONS[topic],
                    "A": to_json(data["A"]),
                }
                if "b" in data:
                    rec["b"] = to_json(data["b"])
                rec["solution"] = _solution_to_json(topic, sol)
                cell.append(rec)
            dev_idx = set(rng.sample(range(PER_CELL), DEV_PER_CELL))  # stratified split
            for k, rec in enumerate(cell):
                rec["split"] = "dev" if k in dev_idx else "test"
            problems.extend(cell)
    return problems


def write(path: Path, seed: int = SEED):
    probs = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every OS so the MD5 matches
        for p in probs:
            f.write(json.dumps(p, separators=(",", ":")) + "\n")
    return probs


def load(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
