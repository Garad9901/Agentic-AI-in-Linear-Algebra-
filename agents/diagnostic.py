"""Diagnostic Agent: first wrong step and a misconception label from data/taxonomy.json."""
from __future__ import annotations

import json
from pathlib import Path

from . import prompts as P
from .base import ask_json

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"


def load_taxonomy(path: Path = TAXONOMY_PATH) -> list:
    taxonomy = json.loads(path.read_text(encoding="utf-8"))["categories"]
    if not isinstance(taxonomy, list):
        raise ValueError('"categories" in data/taxonomy.json must be a list')
    return taxonomy


def _labels(taxonomy: list) -> list:
    return [t["label"] if isinstance(t, dict) else t for t in taxonomy]


def diagnose(problem: dict, taxonomy: list, reference: str | None = None, check_result: str | None = None) -> dict:
    """Returns {"first_wrong_step": k or None (1-based), "misconception": label or None}.

    An empty taxonomy gives misconception None: a label is never invented.
    """
    labels = _labels(taxonomy)
    reply = ask_json("diagnostic", P.diagnostic(problem, labels, reference, check_result)) or {}
    n_steps = len(problem["student_steps"])
    step = reply.get("first_wrong_step")
    if not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= n_steps:
        step = None
    label = reply.get("misconception")
    return {"first_wrong_step": step, "misconception": label if label in labels else None}
