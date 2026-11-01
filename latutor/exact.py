"""Safe exact parsing.

LLM and transcription output is untrusted, so nothing here calls sympify/eval.
Accepted scalar forms: integers ("-3"), fractions ("7/4", "-7/4"), finite
decimals ("0.25", parsed exactly as 1/4). Anything else raises ParseError.
"""
from __future__ import annotations

import re
from fractions import Fraction

import sympy as sp

_INT = re.compile(r"^[+-]?\d+$")
_FRAC = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_DEC = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)$")


class ParseError(ValueError):
    pass


def num(x) -> sp.Rational:
    """Parse one scalar to an exact sympy Rational."""
    if isinstance(x, bool):
        raise ParseError(f"boolean is not a number: {x!r}")
    if isinstance(x, int):
        return sp.Integer(x)
    if isinstance(x, sp.Rational):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise ParseError("non-finite float")
        # JSON floats: take the shortest decimal representation exactly
        return num(repr(x))
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if not isinstance(x, str):
        raise ParseError(f"cannot parse {type(x).__name__} as a number")
    s = x.strip().replace("−", "-").replace(" ", "")
    if _INT.match(s):
        return sp.Integer(int(s))
    m = _FRAC.match(s)
    if m:
        p, q = int(m.group(1)), int(m.group(2))
        if q == 0:
            raise ParseError("zero denominator")
        return sp.Rational(p, q)
    if _DEC.match(s):
        f = Fraction(s)
        return sp.Rational(f.numerator, f.denominator)
    raise ParseError(f"unsupported number format: {x!r}")


def vec(v) -> sp.Matrix:
    """Parse a vector (list of scalars, or n x 1 / 1 x n nested list) to a column Matrix."""
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        raise ParseError("vector must be a non-empty list")
    if all(isinstance(e, (list, tuple)) for e in v):
        flat = [e for row in v for e in row]
        if not (len(v) == 1 or all(len(row) == 1 for row in v)):
            raise ParseError("nested vector must be a single row or a single column")
        v = flat
    return sp.Matrix([num(e) for e in v])


def mat(m) -> sp.Matrix:
    """Parse a rectangular matrix given as a list of rows."""
    if not isinstance(m, (list, tuple)) or len(m) == 0:
        raise ParseError("matrix must be a non-empty list of rows")
    if not all(isinstance(r, (list, tuple)) for r in m):
        raise ParseError("matrix rows must be lists")
    w = len(m[0])
    if w == 0 or any(len(r) != w for r in m):
        raise ParseError("matrix is not rectangular")
    return sp.Matrix([[num(e) for e in r] for r in m])


def to_json(x):
    """Exact sympy object -> JSON-safe (ints stay ints, other rationals become 'p/q')."""
    if isinstance(x, sp.MatrixBase):
        if x.cols == 1:
            return [to_json(e) for e in x]
        return [[to_json(e) for e in x.row(i)] for i in range(x.rows)]
    if isinstance(x, (list, tuple)):
        return [to_json(e) for e in x]
    if isinstance(x, dict):
        return {str(k): to_json(v) for k, v in x.items()}
    if isinstance(x, (bool, str)) or x is None:
        return x
    r = sp.Rational(x)
    return int(r) if r.q == 1 else f"{r.p}/{r.q}"


def same_rowspace(M: sp.Matrix, N: sp.Matrix) -> bool:
    """True if M and N (same width) are row-equivalent, i.e. have the same row space."""
    if M.cols != N.cols:
        return False
    rm, rn = M.rank(), N.rank()
    return rm == rn and M.col_join(N).rank() == rm


def is_rref(R: sp.Matrix) -> bool:
    """True if R is in reduced row echelon form."""
    last_pivot = -1
    seen_zero_row = False
    for i in range(R.rows):
        row = R.row(i)
        nz = [j for j in range(R.cols) if row[j] != 0]
        if not nz:
            seen_zero_row = True
            continue
        if seen_zero_row:
            return False  # non-zero row below a zero row
        p = nz[0]
        if p <= last_pivot or row[p] != 1:
            return False
        if any(R[k, p] != 0 for k in range(R.rows) if k != i):
            return False
        last_pivot = p
    return True
