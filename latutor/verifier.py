"""Verification Agent (SymPy only, no LLM).

Checks one claim at a time against an exact certificate computed from the
problem data (A, and b for least squares). It never reads the stored solution:
the ground truth is used only by grader.py (circularity guard, PREREGISTRATION.md s3).

Every check returns a Verdict:
  certified   - the certificate holds exactly
  refuted     - the certificate fails exactly
  uncheckable - the claim type has no certificate, or its values could not be parsed

Claim schema (shared by every system and by student-step transcriptions)
------------------------------------------------------------------------
Every claim is a JSON object with a "type" and fields below. Scalars may be
integers, "p/q" strings or finite decimals. Matrices are lists of rows.

  rank                     {"value": int}
  rref                     {"matrix": M}
  pivot_columns            {"value": [int, ...]}                    (1-indexed)
  determinant              {"value": q}
  inverse                  {"matrix": M}
  eigenvalue               {"value": q, "algebraic_multiplicity": int (optional)}
  eigenvalues              {"values": [{"value": q, "algebraic_multiplicity": int}, ...]}
  eigenvector              {"eigenvalue": q, "vector": v}
  eigenspace_basis         {"eigenvalue": q, "vectors": [v, ...]}
  characteristic_polynomial{"coefficients": [q, ...]}  highest degree first, det(lam I - A) or det(A - lam I)
  diagonalisable           {"value": bool}
  diagonalisation          {"P": M, "D": M}
  normal_equations         {"AtA": M (optional), "Atb": v (optional)}
  lsq_solution             {"vector": v}
  intermediate_matrix      {"matrix": M, "context": one of
                               "A" | "augmented_A_I" | "A_minus_lambda_I" | "AtA" | "normal_augmented",
                            "lambda": q (only for A_minus_lambda_I)}
  det_relation             {"coefficient": q, "matrix": M}           claims det(A) = coefficient * det(M)
  det_combination          {"terms": [{"coefficient": q, "matrix": M}, ...]}  det(A) = sum coef_i det(M_i)
  statement                {"text": str}                             free text: uncheckable
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import sympy as sp

from .exact import ParseError, is_rref, mat, num, same_rowspace, vec

LAM = sp.Symbol("lam")


@dataclass
class Verdict:
    status: str  # certified | refuted | uncheckable
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def _ok(cond, why_fail, why_ok="certificate holds"):
    return Verdict("certified", why_ok) if cond else Verdict("refuted", why_fail)


class Verifier:
    """Certificate checks for one problem. Results that depend only on A are cached."""

    def __init__(self, problem: dict):
        self.problem = problem
        self.topic = problem.get("topic")
        self.A = mat(problem["A"])
        self.b = vec(problem["b"]) if problem.get("b") is not None else None
        self._cache = {}

    # -- cached exact quantities derived from A (never from the stored solution) --
    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def n(self):
        return self.A.rows

    def charpoly(self):
        return self._get("cp", lambda: sp.Poly(self.A.charpoly(LAM).as_expr(), LAM))

    def det(self):
        return self._get("det", lambda: self.A.det(method="bareiss"))

    def rank(self):
        return self._get("rank", lambda: self.A.rank())

    def rref(self):
        return self._get("rref", lambda: self.A.rref())

    def _root_mult(self, lam):
        p, k = self.charpoly(), 0
        while p.degree() > 0 and p.eval(lam) == 0:
            p = sp.Poly(sp.quo(p.as_expr(), LAM - lam, LAM), LAM)
            k += 1
        return k

    def reference(self, context, lam=None):
        A = self.A
        if context == "A":
            return A
        if context == "augmented_A_I":
            return A.row_join(sp.eye(self.n))
        if context == "A_minus_lambda_I":
            return A - num(lam) * sp.eye(self.n)
        if context in ("AtA", "normal_augmented"):
            if self.b is None:
                raise ParseError("least-squares context needs b")
            AtA = A.T * A
            return AtA if context == "AtA" else AtA.row_join(A.T * self.b)
        raise ParseError(f"unknown context {context!r}")

    # -- main entry point --
    def check(self, claim: dict) -> Verdict:
        if not isinstance(claim, dict) or "type" not in claim:
            return Verdict("uncheckable", "claim is not an object with a type")
        t = claim["type"]
        fn = getattr(self, "_c_" + str(t), None)
        if fn is None:
            return Verdict("uncheckable", f"no certificate for claim type {t!r}")
        try:
            return fn(claim)
        except ParseError as e:
            return Verdict("uncheckable", f"malformed claim: {e}")
        except Exception as e:  # never let one bad claim crash a run
            return Verdict("uncheckable", f"malformed claim: {type(e).__name__}: {e}")

    # -- certificates --
    def _c_rank(self, c):
        return _ok(int(c["value"]) == self.rank(), "rank(A) differs")

    def _c_rref(self, c):
        R = mat(c["matrix"])
        if R.shape != self.A.shape:
            return Verdict("refuted", "shape differs from A")
        if not is_rref(R):
            return Verdict("refuted", "matrix is not in RREF")
        return _ok(same_rowspace(self.A, R), "not row-equivalent to A")

    def _c_pivot_columns(self, c):
        claimed = [int(x) for x in c["value"]]
        return _ok(claimed == [p + 1 for p in self.rref()[1]], "pivot columns differ")

    def _c_determinant(self, c):
        return _ok(num(c["value"]) == self.det(), "det(A) differs")

    def _c_inverse(self, c):
        B = mat(c["matrix"])
        if B.shape != self.A.shape:
            return Verdict("refuted", "shape differs from A")
        return _ok(self.A * B == sp.eye(self.n), "A * B != I")

    def _c_eigenvalue(self, c):
        lam = num(c["value"])
        k = self._root_mult(lam)
        if k == 0:
            return Verdict("refuted", "det(A - lam I) != 0")
        m = c.get("algebraic_multiplicity")
        if m is not None and int(m) != k:
            return Verdict("refuted", f"algebraic multiplicity is {k}")
        return Verdict("certified", "root of the characteristic polynomial")

    def _c_eigenvalues(self, c):
        vals = c["values"]
        total = 0
        seen = set()
        for item in vals:
            if isinstance(item, dict):
                lam, m = num(item["value"]), int(item.get("algebraic_multiplicity", 1))
            else:
                lam, m = num(item), 1
            if lam in seen:
                continue
            seen.add(lam)
            k = self._root_mult(lam)
            if k == 0:
                return Verdict("refuted", f"{lam} is not an eigenvalue")
            if isinstance(item, dict) and "algebraic_multiplicity" in item and m != k:
                return Verdict("refuted", f"multiplicity of {lam} is {k}")
            total += k
        return _ok(total == self.n, "list is incomplete", "complete list of roots")

    def _c_eigenvector(self, c):
        lam, v = num(c["eigenvalue"]), vec(c["vector"])
        if v.rows != self.n:
            return Verdict("refuted", "wrong length")
        if v == sp.zeros(self.n, 1):
            return Verdict("refuted", "zero vector")
        return _ok((self.A - lam * sp.eye(self.n)) * v == sp.zeros(self.n, 1), "A v != lam v")

    def _c_eigenspace_basis(self, c):
        lam = num(c["eigenvalue"])
        V = [vec(v) for v in c["vectors"]]
        if not V or any(v.rows != self.n for v in V):
            return Verdict("refuted", "empty or wrong length")
        M = self.A - lam * sp.eye(self.n)
        if any(M * v != sp.zeros(self.n, 1) for v in V):
            return Verdict("refuted", "a vector is not in the eigenspace")
        Vm = sp.Matrix.hstack(*V)
        if Vm.rank() != len(V):
            return Verdict("refuted", "vectors are linearly dependent")
        return _ok(len(V) == self.n - M.rank(), "does not span the eigenspace")

    def _c_characteristic_polynomial(self, c):
        coeffs = [num(x) for x in c["coefficients"]]
        true = self.charpoly().all_coeffs()
        neg = [((-1) ** self.n) * x for x in true]
        return _ok(coeffs == list(true) or coeffs == neg, "coefficients differ")

    def _c_diagonalisable(self, c):
        val = c["value"]
        if not isinstance(val, bool):
            raise ParseError("value must be true/false")
        return _ok(val == self._get("isdiag", self.A.is_diagonalizable), "diagonalisability differs")

    def _c_diagonalisation(self, c):
        P, D = mat(c["P"]), mat(c["D"])
        if P.shape != self.A.shape or D.shape != self.A.shape:
            return Verdict("refuted", "shape differs from A")
        if not D.is_diagonal():
            return Verdict("refuted", "D is not diagonal")
        if P.det() == 0:
            return Verdict("refuted", "P is singular")
        return _ok(self.A * P == P * D, "A P != P D")

    def _c_normal_equations(self, c):
        if self.b is None:
            return Verdict("uncheckable", "problem has no b")
        if "AtA" not in c and "Atb" not in c:
            raise ParseError("needs AtA or Atb")
        if "AtA" in c and mat(c["AtA"]) != self.A.T * self.A:
            return Verdict("refuted", "A^T A differs")
        if "Atb" in c and vec(c["Atb"]) != self.A.T * self.b:
            return Verdict("refuted", "A^T b differs")
        return Verdict("certified", "normal equations match")

    def _c_lsq_solution(self, c):
        if self.b is None:
            return Verdict("uncheckable", "problem has no b")
        x = vec(c["vector"])
        if x.rows != self.A.cols:
            return Verdict("refuted", "wrong length")
        return _ok(self.A.T * (self.A * x - self.b) == sp.zeros(self.A.cols, 1),
                   "A^T (A x - b) != 0")

    def _c_intermediate_matrix(self, c):
        M = mat(c["matrix"])
        ref = self.reference(c.get("context", "A"), c.get("lambda"))
        if M.shape != ref.shape:
            return Verdict("refuted", "shape differs from the reference matrix")
        return _ok(same_rowspace(ref, M), "not row-equivalent to the reference matrix")

    def _c_det_relation(self, c):
        M = mat(c["matrix"])
        if not M.is_square:
            return Verdict("refuted", "matrix is not square")
        return _ok(num(c["coefficient"]) * M.det(method="bareiss") == self.det(),
                   "det(A) != coefficient * det(M)")

    def _c_det_combination(self, c):
        total = 0
        for term in c["terms"]:
            M = mat(term["matrix"])
            if not M.is_square:
                return Verdict("refuted", "a matrix is not square")
            total += num(term["coefficient"]) * M.det(method="bareiss")
        return _ok(total == self.det(), "det(A) != sum of terms")

    def _c_statement(self, c):
        return Verdict("uncheckable", "free-text statement")

    # -- the Verification Agent's own answer, computed from A (used by S0's hint leak guard) --
    def answer_claims(self) -> list[dict]:
        from .exact import to_json
        t, A = self.topic, self.A
        if t == "rref":
            R, piv = self.rref()
            return [{"type": "rref", "matrix": to_json(R)}, {"type": "rank", "value": self.rank()}]
        if t == "det":
            return [{"type": "determinant", "value": to_json(self.det())}]
        if t == "inverse":
            return [{"type": "inverse", "matrix": to_json(A.inv())}]
        if t in ("eigen", "diag"):
            ev = sorted(A.eigenvals().items(), key=lambda kv: kv[0])
            return [{"type": "eigenvalues",
                     "values": [{"value": to_json(l), "algebraic_multiplicity": int(m)} for l, m in ev]}]
        if t == "lsq":
            x = (A.T * A).LUsolve(A.T * self.b)
            return [{"type": "lsq_solution", "vector": to_json(x)}]
        return []

    # -- student work --
    def check_steps(self, steps: list[dict]) -> dict:
        """Check every step against A. The first refuted step is the predicted first error."""
        verdicts = [self.check(s) for s in steps]
        first = next((i for i, v in enumerate(verdicts) if v.status == "refuted"), None)
        return {
            "verdicts": [v.to_dict() for v in verdicts],
            "first_error_index": first,  # 0-based index into steps, or None
            "n_certified": sum(v.status == "certified" for v in verdicts),
            "n_refuted": sum(v.status == "refuted" for v in verdicts),
            "n_uncheckable": sum(v.status == "uncheckable" for v in verdicts),
        }


def check_claims(problem: dict, claims: list[dict]) -> list[dict]:
    v = Verifier(problem)
    return [v.check(c).to_dict() for c in claims]
