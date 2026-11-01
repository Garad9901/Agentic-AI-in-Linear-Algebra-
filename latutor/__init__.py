"""Verification-guided agentic AI tutor for linear algebra (conference version).

Modules
-------
exact      : safe parsing of numbers, vectors and matrices into exact SymPy objects
generator  : certified problem bank D1 (PREREGISTRATION.md section 4.1)
verifier   : Verification Agent (SymPy, no LLM) - certificate checks on claims and student steps
grader     : ground-truth grader (compares claims with the generator's stored solution)
llm        : LLM backends (OpenAI-compatible HTTP, plus a mock for dry runs only)
agents     : Solver, Diagnostic, Hint and LLM-verifier agents
systems    : S0, B1-B4 and ablations A1-A4 (PREREGISTRATION.md section 5)
leakage    : automated answer-leakage detector for hints
stats      : McNemar, matched-pairs odds ratio, Holm, Wilcoxon, clustered bootstrap
"""
__version__ = "0.1.0"
