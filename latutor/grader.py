"""Ground-truth grader (PREREGISTRATION.md section 3, fixed rule: no LLM judge).

Grades a claim as correct / wrong / unscorable by comparing it with the stored
solution that the generator computed before any tutor ran. It never calls the
Verification Agent, so the agent never grades itself.

Some claim types (intermediate matrices, determinant relations, eigenvectors)
admit many correct answers; they are compared with the stored solution through
the invariant they must share with it (row space, null space, determinant value).
"""
from __future__ import annotations

import sympy as sp

from .exact import ParseError, mat, num, same_rowspace, vec


class Grader:
    def __init__(self, problem: dict):
        self.p = problem
        self.s = problem["solution"]
        self.topic = problem["topic"]
        self.A = mat(problem["A"])  # problem data (shapes, reference matrices), not the answer
        self.n = self.A.rows

    def _eig_mult(self):
        return {num(e["value"]): int(e["algebraic_multiplicity"]) for e in self.s["eigenvalues"]}

    def _eig_space(self, lam):
        for e in self.s.get("eigenspaces", []):
            if num(e["eigenvalue"]) == lam:
                return [vec(b) for b in e["basis"]]
        return None

    def _in_span(self, v, basis):
        B = sp.Matrix.hstack(*basis)
        return B.row_join(v).rank() == B.rank()

    def grade(self, claim: dict) -> str:
        if not isinstance(claim, dict):
            return "unscorable"
        fn = getattr(self, "_g_" + str(claim.get("type")), None)
        if fn is None:
            return "unscorable"
        try:
            r = fn(claim)
        except (ParseError, KeyError):
            return "unscorable"
        except Exception:
            return "unscorable"
        if r is None:
            return "unscorable"
        return "correct" if r else "wrong"

    # ---- final-answer types ----
    def _g_rank(self, c):
        return int(c["value"]) == self.s["rank"] if "rank" in self.s else None

    def _g_rref(self, c):
        return mat(c["matrix"]) == mat(self.s["rref"]) if "rref" in self.s else None

    def _g_pivot_columns(self, c):
        return [int(x) for x in c["value"]] == self.s["pivot_columns"] if "pivot_columns" in self.s else None

    def _g_determinant(self, c):
        return num(c["value"]) == num(self.s["determinant"]) if "determinant" in self.s else None

    def _g_inverse(self, c):
        return mat(c["matrix"]) == mat(self.s["inverse"]) if "inverse" in self.s else None

    def _g_eigenvalue(self, c):
        if "eigenvalues" not in self.s:
            return None
        em, lam = self._eig_mult(), num(c["value"])
        if lam not in em:
            return False
        m = c.get("algebraic_multiplicity")
        return m is None or int(m) == em[lam]

    def _g_eigenvalues(self, c):
        if "eigenvalues" not in self.s:
            return None
        claimed = {}
        for item in c["values"]:
            if isinstance(item, dict):
                claimed[num(item["value"])] = int(item.get("algebraic_multiplicity", 1))
            else:
                lam = num(item)
                claimed[lam] = claimed.get(lam, 0) + 1
        return claimed == self._eig_mult()

    def _g_eigenvector(self, c):
        if "eigenspaces" not in self.s:
            return None
        lam, v = num(c["eigenvalue"]), vec(c["vector"])
        basis = self._eig_space(lam)
        if basis is None or v.rows != self.n or v == sp.zeros(self.n, 1):
            return False
        return self._in_span(v, basis)

    def _g_eigenspace_basis(self, c):
        if "eigenspaces" not in self.s:
            return None
        lam = num(c["eigenvalue"])
        basis = self._eig_space(lam)
        V = [vec(v) for v in c["vectors"]]
        if basis is None or not V or any(v.rows != self.n for v in V):
            return False
        if len(V) != len(basis) or sp.Matrix.hstack(*V).rank() != len(V):
            return False
        return all(self._in_span(v, basis) for v in V)

    def _g_characteristic_polynomial(self, c):
        if "characteristic_polynomial" not in self.s:
            return None
        coeffs = [num(x) for x in c["coefficients"]]
        gold = [num(x) for x in self.s["characteristic_polynomial"]]
        return coeffs == gold or coeffs == [((-1) ** self.n) * x for x in gold]

    def _g_diagonalisable(self, c):
        if "diagonalisable" not in self.s or not isinstance(c["value"], bool):
            return None
        return c["value"] == self.s["diagonalisable"]

    def _g_diagonalisation(self, c):
        if "eigenspaces" not in self.s:
            return None
        P, D = mat(c["P"]), mat(c["D"])
        if P.shape != (self.n, self.n) or D.shape != (self.n, self.n) or not D.is_diagonal():
            return False
        diag = [D[i, i] for i in range(self.n)]
        gold = sorted([lam for lam, m in self._eig_mult().items() for _ in range(m)])
        if sorted(diag) != gold or P.det() == 0:
            return False
        for j in range(self.n):
            basis = self._eig_space(diag[j])
            if basis is None or not self._in_span(P.col(j), basis):
                return False
        return True

    def _g_normal_equations(self, c):
        if "AtA" not in self.s:
            return None
        if "AtA" not in c and "Atb" not in c:
            return None
        ok = True
        if "AtA" in c:
            ok &= mat(c["AtA"]) == mat(self.s["AtA"])
        if "Atb" in c:
            ok &= vec(c["Atb"]) == vec(self.s["Atb"])
        return ok

    def _g_lsq_solution(self, c):
        return vec(c["vector"]) == vec(self.s["x_hat"]) if "x_hat" in self.s else None

    # ---- intermediate work ----
    def _g_intermediate_matrix(self, c):
        M = mat(c["matrix"])
        ctx = c.get("context", "A")
        s = self.s
        if ctx == "A":
            if "rref" in s:
                gold = mat(s["rref"])
            elif "inverse" in s:  # invertible: row space is everything
                gold = sp.eye(self.n)
            elif "determinant" in s and num(s["determinant"]) != 0:
                gold = sp.eye(self.n)
            else:
                return None
            return M.shape == gold.shape and same_rowspace(M, gold)
        if ctx == "augmented_A_I" and "inverse" in s:
            gold = sp.eye(self.n).row_join(mat(s["inverse"]))  # RREF of [A | I]
            return M.shape == gold.shape and same_rowspace(M, gold)
        if ctx == "A_minus_lambda_I" and "eigenvalues" in s:
            lam = num(c["lambda"])
            if M.shape != (self.n, self.n):
                return False
            basis = self._eig_space(lam) or []
            if any(M * b != sp.zeros(self.n, 1) for b in basis):
                return False
            return M.rank() == self.n - len(basis)
        if ctx == "AtA" and "AtA" in s:
            gold = mat(s["AtA"])
            return M.shape == gold.shape and same_rowspace(M, gold)
        if ctx == "normal_augmented" and "AtA" in s:
            gold = mat(s["AtA"]).row_join(vec(s["Atb"]))
            return M.shape == gold.shape and same_rowspace(M, gold)
        return None

    def _g_det_relation(self, c):
        if "determinant" not in self.s:
            return None
        M = mat(c["matrix"])
        if not M.is_square:
            return False
        return num(c["coefficient"]) * M.det(method="bareiss") == num(self.s["determinant"])

    def _g_det_combination(self, c):
        if "determinant" not in self.s:
            return None
        total = 0
        for t in c["terms"]:
            M = mat(t["matrix"])
            if not M.is_square:
                return False
            total += num(t["coefficient"]) * M.det(method="bareiss")
        return total == num(self.s["determinant"])


def grade_claims(problem: dict, claims: list[dict]) -> list[str]:
    g = Grader(problem)
    return [g.grade(c) for c in claims]


def gold_final_claims(problem: dict) -> list[dict]:
    """The stored solution expressed as final-answer claims (used for tests and leak checks)."""
    s, t = problem["solution"], problem["topic"]
    if t == "rref":
        return [{"type": "rref", "matrix": s["rref"]}, {"type": "rank", "value": s["rank"]},
                {"type": "pivot_columns", "value": s["pivot_columns"]}]
    if t == "det":
        return [{"type": "determinant", "value": s["determinant"]}]
    if t == "inverse":
        return [{"type": "inverse", "matrix": s["inverse"]}]
    if t in ("eigen", "diag"):
        out = [{"type": "eigenvalues", "values": s["eigenvalues"]},
               {"type": "characteristic_polynomial", "coefficients": s["characteristic_polynomial"]}]
        out += [{"type": "eigenspace_basis", "eigenvalue": e["eigenvalue"], "vectors": e["basis"]}
                for e in s["eigenspaces"]]
        if t == "diag":
            out.append({"type": "diagonalisation", "P": s["P"], "D": s["D"]})
        return out
    if t == "lsq":
        return [{"type": "normal_equations", "AtA": s["AtA"], "Atb": s["Atb"]},
                {"type": "lsq_solution", "vector": s["x_hat"]}]
    raise ValueError(t)
