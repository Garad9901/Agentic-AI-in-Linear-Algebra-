"""Automated answer-leakage detector for hints (RQ3 primary metric, automated estimate).

A hint leaks if
  (a) it carries a structured claim of a final-answer type that is correct for the problem, or
  (b) its text states the final answer in a recognisable form (the determinant value; all entries of
      x_hat; most fractional entries of the inverse; the full eigenvalue list after "eigenvalues are";
      "rank is r").

The "value that determines it in one step" part of the pre-registered definition needs human judgement,
so the blind human rating (rubric criterion "does not give away the answer") is reported alongside.

`judge(claim) -> bool` decides whether a claim is correct. For the reported metric it is the ground-truth
grader; inside S0's leak guard it is the Verification Agent (which never sees the stored solution).
"""
from __future__ import annotations

import re

from .exact import ParseError, num

FINAL_TYPES = {
    "rref": {"rref", "rank"},
    "det": {"determinant"},
    "inverse": {"inverse"},
    "eigen": {"eigenvalues"},
    "diag": {"eigenvalues", "diagonalisation"},
    "lsq": {"lsq_solution"},
}

_NUM = re.compile(r"(?<![\w/.])[-−]?\d+(?:\s*/\s*\d+)?(?![\w/.])")


def _numbers_in(text):
    out = set()
    for m in _NUM.finditer(text or ""):
        try:
            out.add(num(m.group(0).replace("−", "-").replace(" ", "")))
        except ParseError:
            pass
    return out


def _values(answer_claims, typ, key):
    for c in answer_claims:
        if c.get("type") == typ:
            return c.get(key)
    return None


def detect(hints: list[dict], topic: str, answer_claims: list[dict], judge) -> dict:
    """Return {"leak": bool, "per_hint": [bool...], "reasons": [...]} for a list of hints."""
    per, reasons = [], []
    for h in hints:
        leak, why = False, []
        for c in h.get("claims") or []:
            if isinstance(c, dict) and c.get("type") in FINAL_TYPES.get(topic, set()):
                try:
                    if judge(c):
                        leak = True
                        why.append(f"correct final-answer claim of type {c['type']}")
                except Exception:
                    pass
        text = h.get("text") or ""
        nums = _numbers_in(text)
        try:
            if topic == "det":
                d = num(_values(answer_claims, "determinant", "value"))
                if d in nums and (abs(d) >= 10 or d.q != 1 or re.search(r"det", text, re.I)):
                    leak = True; why.append("determinant value in text")
            elif topic == "inverse":
                M = _values(answer_claims, "inverse", "matrix")
                fr = {num(e) for row in M for e in row if num(e).q != 1}
                if fr and len(fr & nums) >= max(2, (len(fr) + 1) // 2):
                    leak = True; why.append("most fractional inverse entries in text")
            elif topic == "lsq":
                x = [num(e) for e in _values(answer_claims, "lsq_solution", "vector")]
                if all(v in nums for v in x):
                    leak = True; why.append("all x_hat entries in text")
            elif topic in ("eigen", "diag"):
                vals = {num(e["value"]) for e in _values(answer_claims, "eigenvalues", "values")}
                m = re.search(r"eigenvalues?\s*(?:of A\s*)?(?:are|is|:|=|equal)\s*(.{0,120})", text, re.I)
                if m and vals <= _numbers_in(m.group(1)):
                    leak = True; why.append("full eigenvalue list in text")
            elif topic == "rref":
                r = int(_values(answer_claims, "rank", "value"))
                if re.search(rf"rank\s*(?:\(A\)\s*)?(?:of A\s*)?(?:is|=|equals)\s*{r}\b", text, re.I):
                    leak = True; why.append("rank stated in text")
        except (TypeError, ParseError, KeyError, ValueError):
            pass
        per.append(leak)
        reasons.append(why)
    return {"leak": any(per), "per_hint": per, "reasons": reasons}
