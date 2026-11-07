"""Checks the mathematical claims in a hint against problem data, exactly.

A hint declares its claims in JSON. Each is certified, refuted or uncheckable.
A hint passes only if every claim is certified and every number in its text is
accounted for, so a value cannot avoid checking by staying out of the claims
list. Numbers count as accounted for if they appear in the problem, in a
certified claim, or are 0 or +-1.

Claim types:
  determinant {value}        rank {value}              trace {value}
  eigenvalue {value}         eigenvector {eigenvalue, vector}
  charpoly {coefficients}    invertible {value: bool}
  row_op {op, matrix, start?}  the result of applying op to A or [A | I], or to `start`,
                             which must itself be row-equivalent to one of those
  row_equivalent {matrix}    AtA {matrix}              Atb {vector}
  minor {row, col, value}    cofactor {row, col, value}   (1-based)
  arithmetic {expr, value}
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from .certificates import LAM, problem_matrices
from .exact import ParseError, evaluate, mat, num, same_rowspace, vec
from .leak import numbers_in
from .steps import BadOp, apply_ops, parse_ops

CERTIFIED, REFUTED, UNCHECKABLE, WEAK = "certified", "refuted", "uncheckable", "weak"
_TRIVIAL = {sp.Integer(0), sp.Integer(1), sp.Integer(-1)}


def _verdict(cond: bool) -> str:
    return CERTIFIED if cond else REFUTED


def _minor(A, c):
    i, j = int(c["row"]) - 1, int(c["col"]) - 1
    if not (0 <= i < A.rows and 0 <= j < A.cols):
        raise ParseError("minor position is outside the matrix")
    return A.minor_submatrix(i, j).det(method="bareiss"), (-1) ** (i + j)


def _shifted(A) -> list:
    """A - lambda I for each rational eigenvalue, the usual start when finding eigenvectors."""
    if not A.is_square:
        return []
    poly = sp.Poly(A.charpoly(LAM).as_expr(), LAM)
    return [A - lam * sp.eye(A.rows) for lam in sp.roots(poly, filter="Q")]


def _check(A, b, c: dict) -> str:
    t = c.get("type")
    if t == "determinant":
        return _verdict(num(c["value"]) == A.det(method="bareiss"))
    if t == "rank":
        return _verdict(num(c["value"]) == A.rank())
    if t == "trace":
        return _verdict(A.is_square and num(c["value"]) == A.trace())
    if t == "eigenvalue":
        return _verdict((A - num(c["value"]) * sp.eye(A.rows)).det(method="bareiss") == 0)
    if t == "eigenvector":
        lam, v = num(c["eigenvalue"]), vec(c["vector"])
        return _verdict(v.rows == A.rows and v != sp.zeros(A.rows, 1) and A * v == lam * v)
    if t == "charpoly":
        coeffs = [num(x) for x in c["coefficients"]]
        true = list(sp.Poly(A.charpoly(LAM).as_expr(), LAM).all_coeffs())
        return _verdict(coeffs == true or coeffs == [(-1) ** A.rows * x for x in true])
    if t == "invertible":
        if not isinstance(c["value"], bool):
            return UNCHECKABLE
        return _verdict(c["value"] == (A.det(method="bareiss") != 0))
    if t == "row_op":
        ops = parse_ops(c["op"])
        if ops is None:
            return UNCHECKABLE
        M = mat(c["matrix"])
        starts = [A, A.row_join(sp.eye(A.rows))]
        if b is not None:
            AtA, Atb = A.T * A, A.T * b
            starts += [AtA, AtA.row_join(Atb)]
        if c.get("start"):
            start = mat(c["start"])
            if not any(same_rowspace(s, start) for s in starts + _shifted(A)):
                return REFUTED  # a start unconnected to the problem certifies nothing
            starts = [start]
        try:
            return _verdict(any(s.shape == M.shape and apply_ops(s, ops) == M for s in starts))
        except BadOp:
            return REFUTED
    if t == "row_equivalent":
        M = mat(c["matrix"])
        if M.shape != A.shape:
            return UNCHECKABLE
        return WEAK if same_rowspace(A, M) else REFUTED  # never certified: no operation was applied
    if t == "AtA":
        return _verdict(mat(c["matrix"]) == A.T * A)
    if t == "Atb":
        return UNCHECKABLE if b is None else _verdict(vec(c["vector"]) == A.T * b)
    if t == "minor":
        return _verdict(num(c["value"]) == _minor(A, c)[0])
    if t == "cofactor":
        m, sign = _minor(A, c)
        return _verdict(num(c["value"]) == sign * m)
    if t == "arithmetic":
        return _verdict(evaluate(c["expr"]) == num(c["value"]))
    return UNCHECKABLE


def check_claim(problem: dict, claim) -> str:
    if not isinstance(claim, dict):
        return UNCHECKABLE
    try:
        A, b = problem_matrices(problem)
        return _check(A, b, claim)
    except (ParseError, KeyError, TypeError, ValueError):
        return UNCHECKABLE


def _claim_numbers(claim: dict) -> list:
    out: list = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k != "type":
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            out.extend(numbers_in(x.replace("*", " ").replace("<-", " ").replace("->", " ")))
        elif not isinstance(x, bool):
            try:
                out.append(num(x))
            except ParseError:
                pass

    walk(claim)
    return out


@dataclass
class HintCheck:
    verified: bool
    n_claims: int = 0
    n_certified: int = 0
    reasons: list = field(default_factory=list)


def verify_hint(problem: dict, text: str, claims) -> HintCheck:
    claims = claims if isinstance(claims, list) else []
    result = HintCheck(verified=True, n_claims=len(claims))
    allowed = set(_TRIVIAL)
    A, b = problem_matrices(problem)
    allowed.update(A)
    if b is not None:
        allowed.update(b)
    for c in claims:
        status = check_claim(problem, c)
        if status == CERTIFIED:
            result.n_certified += 1
            allowed.update(_claim_numbers(c))
        else:
            result.verified = False
            result.reasons.append(f"claim {status}: {c}")
    unattested = [x for x in numbers_in(text) if x not in allowed]
    if unattested:
        result.verified = False
        result.reasons.append(f"values in the text with no certified claim: {sorted(set(map(str, unattested)))}")
    return result
