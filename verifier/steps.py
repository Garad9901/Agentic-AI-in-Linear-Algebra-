"""Step checker: the first step that is not a valid consequence of the previous one.

A step is a bare matrix, a bare value, or an object with any of:
  text, op, matrix, vector, value, expr, factor, eigenvalue, coefficients

Row-equivalence alone cannot catch an arithmetic slip when the matrix is square
and full rank, because every invertible n x n matrix is row-equivalent to every
other. So a stated row operation ("R2 <- R2 - 3R1", in `op` or in `text`) is
applied exactly and the result compared. A full-rank square step with no stated
operation passes on row-equivalence alone and is listed in `weak`, so reports
can show how much rests on that weaker check.

Steps that say nothing checkable are skipped and listed in `unchecked`. They are
never reported as wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy as sp

from .certificates import LAM, problem_matrices
from .exact import ParseError, evaluate, mat, num, same_rowspace, unify_minus, vec

_ROW = r"[Rr]_?\{?(\d+)\}?"
_ARROW = r"(?:<-+|←|-+>|→|:=|=|\\to|\\rightarrow|\\leftarrow)"
_SWAP = re.compile(rf"^(?:swap\s+)?{_ROW}\s*(?:<-+>|↔|⇄|\\leftrightarrow|and|with)\s*{_ROW}$", re.I)
_ASSIGN = re.compile(rf"^{_ROW}\s*{_ARROW}\s*(.+)$")
_ASSIGN_RTL = re.compile(rf"^(.+?)\s*(?:-+>|→|\\to|\\rightarrow)\s*{_ROW}$")
_TERM = re.compile(
    r"\s*([+-]?)\s*(?:\(?\s*([+-]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*\)?\s*\*?\s*)?"
    + _ROW + r"(?:\s*/\s*(\d+))?\s*")
_OP_IN_TEXT = re.compile(rf"(?:swap\s+)?{_ROW}\s*(?:<-+>|↔|\\leftrightarrow|{_ARROW}|and\s+{_ROW}).*", re.I)
_DET_WORD = re.compile(r"\bdet", re.I)
_BAD_STEP = (ParseError, TypeError, ValueError, KeyError, IndexError)  # model output is untrusted


class BadOp(ValueError):
    """The operation parsed, but it is not a legal row operation."""


@dataclass
class StepReport:
    first_bad: int | None = None  # 0-based index, or None
    reason: str = ""
    unchecked: list = field(default_factory=list)
    weak: list = field(default_factory=list)


def _normalise_op(s: str) -> str:
    s = unify_minus(s).replace("×", "*").replace("·", "*").replace("$", "")
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", s)
    return s.strip()


def _parse_rhs(s: str):
    pos, terms = 0, []
    while pos < len(s):
        m = _TERM.match(s, pos)
        if not m or m.end() == pos:
            return None
        sign, coef, row, div = m.groups()
        c = num(coef.replace(" ", "")) if coef else sp.Integer(1)
        if sign == "-":
            c = -c
        if div:
            if int(div) == 0:
                return None
            c = c / int(div)
        terms.append((int(row) - 1, c))
        pos = m.end()
    return terms or None


def parse_ops(text) -> list | None:
    """Parse row operations. Returns [("swap", i, j) | ("set", target, terms), ...], or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    ops = []
    for part in re.split(r"[,;]", _normalise_op(text)):
        part = part.strip()
        if not part:
            continue
        m = _SWAP.match(part)
        if m:
            ops.append(("swap", int(m.group(1)) - 1, int(m.group(2)) - 1))
            continue
        m = _ASSIGN.match(part)
        if m:
            target, rhs = int(m.group(1)) - 1, m.group(2)
        else:
            m = _ASSIGN_RTL.match(part)
            if not m:
                return None
            target, rhs = int(m.group(2)) - 1, m.group(1)
        try:
            terms = _parse_rhs(rhs)
        except ParseError:
            return None
        if terms is None:
            return None
        ops.append(("set", target, terms))
    return ops or None


def apply_ops(M: sp.Matrix, ops: list) -> sp.Matrix:
    M = M.copy()
    for op in ops:
        if op[0] == "swap":
            _, i, j = op
            if not (0 <= i < M.rows and 0 <= j < M.rows):
                raise BadOp("the operation names a row that does not exist")
            M.row_swap(i, j)
            continue
        _, target, terms = op
        if not 0 <= target < M.rows or any(not 0 <= k < M.rows for k, _ in terms):
            raise BadOp("the operation names a row that does not exist")
        if sum(c for k, c in terms if k == target) == 0:
            raise BadOp("the operation removes the target row, so it is not invertible")
        new = sp.zeros(1, M.cols)
        for k, c in terms:
            new += c * M.row(k)
        M[target, :] = new
    return M


def _step_ops(step: dict):
    if step.get("op"):
        return parse_ops(step["op"])
    text = step.get("text")
    if isinstance(text, str) and text.strip():
        m = _OP_IN_TEXT.search(text.splitlines()[0])
        return parse_ops(m.group(0).rstrip(".")) if m else None
    return None


def _transition(prev: sp.Matrix, cur: sp.Matrix, step: dict):
    """Returns (reason or None, weak)."""
    if prev.shape != cur.shape:
        return f"shape {cur.shape} does not follow from the previous {prev.shape}", False
    ops = _step_ops(step)
    if ops:
        try:
            expected = apply_ops(prev, ops)
        except BadOp as e:
            return str(e), False
        if expected != cur:
            return "the stated row operation does not give this matrix (wrong arithmetic)", False
        return None, False
    if not same_rowspace(prev, cur):
        return "not row-equivalent to the previous step", False
    return None, cur.rank() == cur.cols


def normalise_step(step) -> dict:
    """Bare matrix/vector/value -> dict. Dict fields that are null or empty are dropped."""
    if isinstance(step, dict):
        return {k: v for k, v in step.items() if v is not None and v != "" and v != []}
    if isinstance(step, (list, tuple)):
        if step and all(isinstance(r, (list, tuple)) for r in step):
            return {"matrix": step}
        return {"vector": step}
    return {"value": step}


def _check_expr(step: dict):
    """Arithmetic in `expr` must evaluate to `value`. Returns a reason or None."""
    if "expr" not in step or "value" not in step:
        return None
    try:
        got = evaluate(step["expr"])
    except ParseError:
        return None
    if got != num(step["value"]):
        return f"arithmetic: {step['expr']} = {got}, not {step['value']}"
    return None


class _Checker:
    def __init__(self, topic: str, problem: dict, steps: list):
        self.topic = topic
        self.A, self.b = problem_matrices(problem)
        self.steps = steps
        self.report = StepReport()
        self.prev = self.A
        self.last_value = max((i for i, s in enumerate(steps) if "value" in s), default=None)
        self.last_matrix = max((i for i, s in enumerate(steps) if "matrix" in s), default=None)
        if topic == "inverse":
            self.prev = self.A.row_join(sp.eye(self.A.rows))
        self.AtA = self.A.T * self.A if self.b is not None else None
        self.Atb = self.A.T * self.b if self.b is not None else None

    def _matrix_step(self, i, s, prev=None):
        """Row-operation transition from the previous matrix. Returns a reason or None."""
        cur = mat(s["matrix"])
        reason, weak = _transition(self.prev if prev is None else prev, cur, s)
        if reason:
            return reason
        if weak:
            self.report.weak.append(i)
        self.prev = cur
        return None

    # Each returns a failure reason, or None.
    def _rref_rank(self, i, s):
        if "matrix" in s:
            return self._matrix_step(i, s)
        if "value" in s:
            if num(s["value"]) != self.prev.rank():
                return f"rank {s['value']} does not follow from the previous matrix"
            return None
        self.report.unchecked.append(i)

    def _determinant(self, i, s):
        if "matrix" in s:
            if "factor" in s:
                cur = mat(s["matrix"])
                if not cur.is_square or num(s["factor"]) * cur.det(method="bareiss") != self.A.det(method="bareiss"):
                    return "det(A) != factor * det(this matrix)"
                self.prev = cur
                return None
            return self._matrix_step(i, s)
        if "value" in s:
            # Only the last value is the determinant; earlier ones may be minors or partial sums.
            if i == self.last_value:
                if num(s["value"]) != self.A.det(method="bareiss"):
                    return f"{s['value']} is not the determinant the previous steps give"
            elif "expr" not in s:
                self.report.unchecked.append(i)
            return None
        self.report.unchecked.append(i)

    def _inverse(self, i, s):
        n = self.A.rows
        if "matrix" in s:
            cur = mat(s["matrix"])
            if cur.shape == (n, 2 * n):
                return self._matrix_step(i, s)
            if cur.shape == (n, n):
                if i != self.last_matrix:
                    self.report.unchecked.append(i)  # side working, e.g. a cofactor matrix
                elif self.A * cur != sp.eye(n):
                    return "A * B != I"
                return None
            return f"shape {cur.shape} does not fit [A | I] or an inverse"
        if "value" in s and _DET_WORD.search(str(s.get("text", ""))):
            if num(s["value"]) != self.A.det(method="bareiss"):
                return "stated determinant is not det(A)"
            return None
        self.report.unchecked.append(i)

    def _eigen(self, i, s):
        A, n = self.A, self.A.rows
        lam = num(s["eigenvalue"]) if "eigenvalue" in s else None
        if "coefficients" in s:
            coeffs = [num(c) for c in s["coefficients"]]
            true = list(sp.Poly(A.charpoly(LAM).as_expr(), LAM).all_coeffs())
            if coeffs != true and coeffs != [(-1) ** n * c for c in true]:
                return "characteristic polynomial coefficients are wrong"
            return None
        if "vector" in s:
            v = vec(s["vector"])
            if v.rows != n:
                self.report.unchecked.append(i)
            elif v == sp.zeros(n, 1):
                return "zero vector is not an eigenvector"
            elif lam is not None:
                if A * v != lam * v:
                    return f"A*v != {lam}*v"
            elif sp.Matrix.hstack(v, A * v).rank() != 1:
                return "A*v is not a multiple of v"
            return None
        if "matrix" in s:
            cur = mat(s["matrix"])
            if lam is None or cur.shape != A.shape:
                self.report.unchecked.append(i)  # P, D or other side working
            elif not same_rowspace(A - lam * sp.eye(n), cur):
                return f"not row-equivalent to A - ({lam})I"
            return None
        if lam is None and "value" in s:
            lam = num(s["value"])
        if lam is None:
            self.report.unchecked.append(i)
        elif (A - lam * sp.eye(n)).det(method="bareiss") != 0:
            return f"{lam} is not a root of the characteristic polynomial"

    def _least_squares(self, i, s):
        n = self.A.cols
        if "matrix" in s:
            cur = mat(s["matrix"])
            if cur.shape == (n, n + 1):
                ref = self.AtA.row_join(self.Atb)
                if not same_rowspace(ref, cur):
                    return "not row-equivalent to [A^T A | A^T b]"
                return self._matrix_step(i, s, self.prev if self.prev.shape == ref.shape else ref)
            if cur.shape == (n, n):
                if cur != self.AtA:
                    return "this is not A^T A"
                return None
            self.report.unchecked.append(i)
            return None
        if "vector" in s:
            v = vec(s["vector"])
            if v.rows != n:
                self.report.unchecked.append(i)
            elif v != self.Atb and self.AtA * v != self.Atb:
                return "neither A^T b nor a solution of the normal equations"
            return None
        self.report.unchecked.append(i)

    def dispatch(self):
        return {"rref_rank": self._rref_rank, "determinant": self._determinant, "inverse": self._inverse,
                "eigenpairs": self._eigen, "diagonalisation": self._eigen,
                "least_squares": self._least_squares}[self.topic]

    def run(self) -> StepReport:
        check = self.dispatch()
        for i, step in enumerate(self.steps):
            try:
                reason = _check_expr(step) or check(i, step)
            except _BAD_STEP:
                self.report.unchecked.append(i)
                continue
            if reason:
                self.report.first_bad = i
                self.report.reason = reason
                return self.report
        return self.report


def step_statuses(topic: str, problem: dict, steps: list) -> list:
    """A status for every step: certified, weak, refuted or uncheckable.

    Unlike check_steps this does not stop at the first bad step. A step after a refuted one
    is judged as a consequence of that step, as a reader would.
    """
    normalised = [normalise_step(s) for s in steps]
    checker = _Checker(topic, problem, normalised)
    check = checker.dispatch()
    refuted = set()
    for i, step in enumerate(normalised):
        try:
            reason = _check_expr(step) or check(i, step)
        except _BAD_STEP:
            checker.report.unchecked.append(i)
            continue
        if reason:
            refuted.add(i)
            try:
                if "matrix" in step:
                    checker.prev = mat(step["matrix"])
            except ParseError:
                pass

    def status(i):
        if i in refuted:
            return "refuted"
        if i in checker.report.unchecked:
            return "uncheckable"
        return "weak" if i in checker.report.weak else "certified"

    return [status(i) for i in range(len(normalised))]


def check_steps(topic: str, problem: dict, steps: list) -> StepReport:
    """Return the first invalid step (0-based), or first_bad=None when none is found."""
    return _Checker(topic, problem, [normalise_step(s) for s in steps]).run()
