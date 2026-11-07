"""Exact certificate checks for final answers. No LLM.

verify() reads only the problem data (A, and b for least squares) and the
candidate. It never reads the stored ground truth, which belongs to grader.py,
so the verifier never grades itself.

Candidate schema, by topic:
  rref_rank        {"rref": M, "rank": r}
  determinant      {"determinant": q}
  inverse          {"inverse": M}
  eigenpairs       {"eigenpairs": [{"eigenvalue": q, "eigenvector": v}, ...]}
                   one entry per eigenvalue counted with algebraic multiplicity
  diagonalisation  {"P": M, "D": M}
  least_squares    {"x": v}
Scalars may be integers, "p/q" strings or finite decimals.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .exact import ParseError, is_rref, mat, num, pivot_count, same_rowspace, vec

LAM = sp.Symbol("lam")


@dataclass
class Verdict:
    ok: bool
    reason: str = ""


def _fail(reason: str) -> Verdict:
    return Verdict(False, reason)


def problem_matrices(problem: dict) -> tuple[sp.Matrix, sp.Matrix | None]:
    A = mat(problem["A"])
    b = vec(problem["b"]) if problem.get("b") is not None else None
    return A, b


def check_rref(A, c) -> Verdict:
    R = mat(c["rref"])
    if R.shape != A.shape:
        return _fail("RREF candidate has a different shape from A")
    if not is_rref(R):
        return _fail("candidate is not in reduced row echelon form")
    if not same_rowspace(A, R):
        return _fail("candidate is not row-equivalent to A")
    r = num(c["rank"])
    if r != pivot_count(R):
        return _fail(f"stated rank {r} differs from the {pivot_count(R)} pivots of the candidate")
    return Verdict(True)


def check_determinant(A, c) -> Verdict:
    if num(c["determinant"]) != A.det(method="bareiss"):
        return _fail("determinant is not exactly det(A)")
    return Verdict(True)


def check_inverse(A, c) -> Verdict:
    B = mat(c["inverse"])
    if B.shape != A.shape:
        return _fail("inverse candidate has a different shape from A")
    if A * B != sp.eye(A.rows):
        return _fail("A * B != I")
    return Verdict(True)


def _multiset_matches(A, values) -> bool:
    """True if the values are the roots of the characteristic polynomial, with multiplicity."""
    target = sp.Poly(A.charpoly(LAM).as_expr(), LAM)
    product = sp.Poly(1, LAM)
    for lam in values:
        product = product * sp.Poly(LAM - lam, LAM)
    return product == target


def check_eigenpairs(A, c) -> Verdict:
    pairs = c["eigenpairs"]
    if not isinstance(pairs, list) or not pairs:
        return _fail("eigenpairs must be a non-empty list")
    n = A.rows
    values = []
    by_value: dict = {}
    for p in pairs:
        lam, v = num(p["eigenvalue"]), vec(p["eigenvector"])
        if v.rows != n:
            return _fail(f"eigenvector for {lam} has the wrong length")
        if v == sp.zeros(n, 1):
            return _fail(f"eigenvector for {lam} is the zero vector")
        if A * v != lam * v:
            return _fail(f"A*v != lambda*v for lambda = {lam}")
        values.append(lam)
        by_value.setdefault(lam, []).append(v)
    if not _multiset_matches(A, values):
        return _fail("eigenvalue multiset does not match the characteristic polynomial")
    # Not in the brief's list, but without it a repeated eigenvector would pass for a repeated eigenvalue.
    for lam, vs in by_value.items():
        if sp.Matrix.hstack(*vs).rank() != len(vs):
            return _fail(f"eigenvectors for lambda = {lam} are linearly dependent")
    return Verdict(True)


def check_diagonalisation(A, c) -> Verdict:
    P, D = mat(c["P"]), mat(c["D"])
    if P.shape != A.shape or D.shape != A.shape:
        return _fail("P or D has a different shape from A")
    # Not in the brief's list, but without it P = I, D = A would pass.
    if not D.is_diagonal():
        return _fail("D is not diagonal")
    if P.det(method="bareiss") == 0:
        return _fail("P is singular")
    if A * P != P * D:
        return _fail("A != P*D*P^-1")
    return Verdict(True)


def check_least_squares(A, b, c) -> Verdict:
    if b is None:
        return _fail("problem has no b")
    x = vec(c["x"])
    if x.rows != A.cols:
        return _fail("x has the wrong length")
    if A.T * A * x != A.T * b:
        return _fail("A^T A x != A^T b")
    return Verdict(True)


def verify(topic: str, problem: dict, candidate) -> Verdict:
    """Check a final answer. Malformed candidates fail with a reason and never raise."""
    if not isinstance(candidate, dict):
        return _fail("answer is not a JSON object")
    try:
        A, b = problem_matrices(problem)
        if topic == "rref_rank":
            return check_rref(A, candidate)
        if topic == "determinant":
            return check_determinant(A, candidate)
        if topic == "inverse":
            return check_inverse(A, candidate)
        if topic == "eigenpairs":
            return check_eigenpairs(A, candidate)
        if topic == "diagonalisation":
            return check_diagonalisation(A, candidate)
        if topic == "least_squares":
            return check_least_squares(A, b, candidate)
    except ParseError as e:
        return _fail(f"malformed answer: {e}")
    except (KeyError, TypeError, IndexError) as e:
        return _fail(f"malformed answer: missing or bad field {e}")
    raise ValueError(f"unknown topic {topic!r}")
