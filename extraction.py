"""Turn one tutor response into the claims it makes.

RQ1 and RQ4 both count claims (PREREGISTRATION.md s3), so both need the same list.
Each entry says where the claim came from:
  {"where": "step" | "final" | "hint", "index": i, "claim": {"type": ..., ...}}

A step yields one claim, the final answer yields one or more, and a hint yields its
declared claims. A step with no computed value is a "statement": it stays in the
denominator of coverage, and is neither right nor wrong.
"""
from __future__ import annotations

import re

from verifier.exact import ParseError, mat
from verifier.steps import normalise_step

_DET_WORD = re.compile(r"\bdet", re.I)


def _statement(step: dict) -> dict:
    return {"type": "statement", "text": str(step.get("text", ""))}


def _shape(matrix):
    try:
        return mat(matrix).shape
    except ParseError:
        return None


def _intermediate(context: str, step: dict, previous: dict, chain: str | None = None, **extra) -> dict:
    chain = chain or context
    claim = {"type": "intermediate_matrix", "context": context, "matrix": step["matrix"], "op": step.get("op"),
             "text": step.get("text"), "previous": previous.get(chain), **extra}
    previous[chain] = step["matrix"]
    return claim


def _rref_step(step, previous):
    if "matrix" in step:
        return _intermediate("A", step, previous)
    if "value" in step:
        return {"type": "rank", "value": step["value"]}
    return _statement(step)


def _determinant_step(step, previous, is_last_value):
    if "matrix" in step:
        if "factor" in step:
            return {"type": "det_relation", "coefficient": step["factor"], "matrix": step["matrix"]}
        return _intermediate("A", step, previous)
    if "value" in step:
        if is_last_value:
            return {"type": "determinant", "value": step["value"]}
        if "expr" in step:
            return {"type": "arithmetic", "expr": step["expr"], "value": step["value"]}
    return _statement(step)


def _inverse_step(step, previous, n, is_last_matrix):
    if "matrix" in step:
        shape = _shape(step["matrix"])
        if shape == (n, 2 * n):
            return _intermediate("augmented_A_I", step, previous)
        if shape == (n, n) and is_last_matrix:
            return {"type": "inverse", "matrix": step["matrix"]}
    elif "value" in step and _DET_WORD.search(str(step.get("text", ""))):
        return {"type": "determinant", "value": step["value"]}
    return _statement(step)


def _eigen_step(step, previous):
    if "coefficients" in step:
        return {"type": "charpoly", "coefficients": step["coefficients"]}
    if "vector" in step and "eigenvalue" in step:
        return {"type": "eigenvector", "eigenvalue": step["eigenvalue"], "vector": step["vector"]}
    if "matrix" in step and "eigenvalue" in step:
        chain = f"lambda={step['eigenvalue']}"  # each eigenvalue has its own reduction
        return _intermediate("A_minus_lambda_I", step, previous, chain, **{"lambda": step["eigenvalue"]})
    if "vector" in step or "matrix" in step:
        return _statement(step)
    if "eigenvalue" in step or "value" in step:
        return {"type": "eigenvalue", "value": step.get("eigenvalue", step.get("value"))}
    return _statement(step)


def _least_squares_step(step, previous, n):
    if "matrix" in step:
        shape = _shape(step["matrix"])
        if shape == (n, n + 1):
            return _intermediate("normal_augmented", step, previous)
        if shape == (n, n):
            return {"type": "AtA", "matrix": step["matrix"]}
    elif "vector" in step and isinstance(step["vector"], list) and len(step["vector"]) == n:
        return {"type": "normal_vector", "vector": step["vector"]}
    return _statement(step)


def _step_claim(problem, step, previous, is_last_value, is_last_matrix) -> dict:
    topic = problem["topic"]
    n = len(problem["A"][0])
    if topic == "rref_rank":
        return _rref_step(step, previous)
    if topic == "determinant":
        return _determinant_step(step, previous, is_last_value)
    if topic == "inverse":
        return _inverse_step(step, previous, n, is_last_matrix)
    if topic in ("eigenpairs", "diagonalisation"):
        return _eigen_step(step, previous)
    return _least_squares_step(step, previous, n)


def final_claims(topic: str, answer) -> list:
    if not isinstance(answer, dict):
        return []
    if topic == "rref_rank":
        return [{"type": "rref", "matrix": answer.get("rref")}, {"type": "rank", "value": answer.get("rank")}]
    if topic == "determinant":
        return [{"type": "determinant", "value": answer.get("determinant")}]
    if topic == "inverse":
        return [{"type": "inverse", "matrix": answer.get("inverse")}]
    if topic == "eigenpairs":
        pairs = answer.get("eigenpairs")
        pairs = [p for p in pairs if isinstance(p, dict)] if isinstance(pairs, list) else []
        values = {"type": "eigenvalues", "values": [p.get("eigenvalue") for p in pairs]}
        return [values] + [{"type": "eigenvector", "eigenvalue": p.get("eigenvalue"), "vector": p.get("eigenvector")}
                           for p in pairs]
    if topic == "diagonalisation":
        return [{"type": "diagonalisation", "P": answer.get("P"), "D": answer.get("D")}]
    return [{"type": "lsq_solution", "vector": answer.get("x")}]


def extract(problem: dict, trace, answer, hints) -> list:
    """Every claim that reached the student in one response."""
    steps = [normalise_step(s) for s in trace] if isinstance(trace, list) else []
    last_value = max((i for i, s in enumerate(steps) if "value" in s), default=None)
    last_matrix = max((i for i, s in enumerate(steps) if "matrix" in s), default=None)
    previous: dict = {}
    claims = []
    for i, step in enumerate(steps):
        claim = _step_claim(problem, step, previous, i == last_value, i == last_matrix)
        claims.append({"where": "step", "index": i, "claim": claim})
    for j, claim in enumerate(final_claims(problem["topic"], answer)):
        claims.append({"where": "final", "index": j, "claim": claim})
    for h, item in enumerate(hints or []):
        for claim in item.get("claims") or []:
            claims.append({"where": "hint", "index": h, "claim": claim})
    return claims
