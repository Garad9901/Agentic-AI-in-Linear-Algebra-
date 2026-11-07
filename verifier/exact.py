"""Safe exact parsing.

LLM and transcription output is untrusted, so nothing here calls sympify or
eval. Accepted scalars: integers ("-3"), fractions ("7/4"), finite decimals
("0.25", parsed exactly as 1/4), and LaTeX fractions. Anything else raises
ParseError.
"""
from __future__ import annotations

import ast
import re
from fractions import Fraction

import sympy as sp

_INT = re.compile(r"^[+-]?\d+$")
_FRAC = re.compile(r"^([+-]?\d+)\s*/\s*([+-]?\d+)$")
_DEC = re.compile(r"^[+-]?(\d+\.\d*|\.\d+)$")
_TEX = re.compile(r"^([+-]?)\\[dt]?frac\{([+-]?\d+)\}\{([+-]?\d+)\}$")
_MINUS = re.compile("[‐‑‒–−]")
_MINUS = re.compile("[‐‑‒–−]")


class ParseError(ValueError):
    pass


def unify_minus(s: str) -> str:
    """Models often write minus as U+2011 or U+2212, which would otherwise read as a plain positive."""
    return _MINUS.sub("-", s)


def unify_minus(s: str) -> str:
    """Models often write minus as U+2011 or U+2212, which would otherwise read as a plain positive."""
    return _MINUS.sub("-", s)


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
        return num(repr(x))  # JSON floats: the shortest decimal representation, exactly
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    if not isinstance(x, str):
        raise ParseError(f"cannot parse {type(x).__name__} as a number")
    s = unify_minus(x).strip().replace(" ", "").strip("$")
    if _INT.match(s):
        return sp.Integer(int(s))
    m = _TEX.match(s)
    if m:
        s = f"{m.group(1)}{m.group(2)}/{m.group(3)}"
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
    """Parse a vector (flat list, or n x 1 / 1 x n nested list) to a column Matrix."""
    if not isinstance(v, (list, tuple)) or len(v) == 0:
        raise ParseError("vector must be a non-empty list")
    if all(isinstance(e, (list, tuple)) for e in v):
        if not (len(v) == 1 or all(len(row) == 1 for row in v)):
            raise ParseError("nested vector must be a single row or a single column")
        v = [e for row in v for e in row]
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


def pivot_count(R: sp.Matrix) -> int:
    return sum(1 for i in range(R.rows) if any(R[i, j] != 0 for j in range(R.cols)))


def _eval_node(node) -> sp.Rational:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return num(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval_node(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ParseError("division by zero")
            return a / b
        if isinstance(node.op, ast.Pow) and b.q == 1 and abs(b) <= 16:
            return a ** int(b)
    raise ParseError("unsupported arithmetic expression")


def evaluate(expr: str) -> sp.Rational:
    """Exactly evaluate +, -, *, /, ** and brackets over numbers. Nothing else is allowed."""
    if not isinstance(expr, str) or len(expr) > 2000:
        raise ParseError("expression must be a short string")
    s = (unify_minus(expr).replace("×", "*").replace("·", "*")
         .replace("÷", "/").replace("^", "**"))
    s = re.sub(r"(?<=[\d)])\s*\(", "*(", s)  # 2(3-1) means 2*(3-1)
    s = re.sub(r"\)\s*(?=\d)", ")*", s)
    try:
        return _eval_node(ast.parse(s.strip(), mode="eval"))
    except SyntaxError:
        raise ParseError(f"cannot parse expression: {expr!r}") from None
