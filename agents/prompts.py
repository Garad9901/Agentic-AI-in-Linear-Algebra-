"""Prompt templates. Tune on the dev split only; `FROZEN` marks them final."""
from __future__ import annotations

import json

NUMBERS = 'Write numbers exactly: integers, or fractions as strings like "-7/4". Never use decimals.'

ANSWER_FORMAT = {
    "rref_rank": '{"rref": [[...], ...], "rank": r}',
    "determinant": '{"determinant": d}',
    "inverse": '{"inverse": [[...], ...]}',
    "eigenpairs": '{"eigenpairs": [{"eigenvalue": l, "eigenvector": [...]}, ...]}',
    "diagonalisation": '{"P": [[...], ...], "D": [[...], ...]}',
    "least_squares": '{"x": [...]}',
}

STEP_FORMAT = {
    "rref_rank": 'one step per row operation: {"text":"...","op":"R2 <- R2 - 3R1","matrix":[[...]]}; '
                 'finish with {"text":"count the pivots","value":r}',
    "determinant": 'row reduction steps {"text":"...","op":"R2 <- R2 - 3R1","matrix":[[...]]}; finish with '
                   '{"text":"...","expr":"2*3*(-1)","value":d} where expr is plain arithmetic',
    "inverse": 'row operations on the augmented matrix [A | I], which has n rows and 2n columns: '
               '{"text":"...","op":"R2 <- R2 - 3R1","matrix":[[...]]}; finish with {"text":"...","matrix":B} for the inverse',
    "eigenpairs": '{"text":"...","coefficients":[...]} for the characteristic polynomial, highest power first; '
                  '{"text":"...","eigenvalue":l} for a root; {"text":"...","eigenvalue":l,"vector":[...]} for an eigenvector',
    "diagonalisation": '{"text":"...","coefficients":[...]} for the characteristic polynomial, highest power first; '
                       '{"text":"...","eigenvalue":l} for a root; {"text":"...","eigenvalue":l,"vector":[...]} for an eigenvector',
    "least_squares": '{"text":"...","matrix":AtA} for A^T A; {"text":"...","vector":Atb} for A^T b; row operations on '
                     '[A^T A | A^T b]: {"text":"...","op":"R2 <- R2 - 3R1","matrix":[[...]]}; finish with {"text":"...","vector":x}',
}

OP_RULES = 'Row operations in "op" are applied exactly by a checker. Use one operation per step, written like ' \
           '"R2 <- R2 - 3R1", "R2 <- (1/2)R2" or "R1 <-> R2".'

SOLUTION_FORMAT = """Reply with one JSON object only, no prose:
{{"steps": [...], "answer": {answer}}}
Each entry of "steps" has this form: {steps}.
{op_rules}
{numbers}"""

HINT_LEVELS = {
    1: "Level 1 (nudge): point to what to look at next. Ask a question. State no computed values.",
    2: "Level 2 (concept): name the concept or rule the student needs and how it applies here.",
    3: "Level 3 (partial worked step): carry out one early step of the working, and stop well before the end.",
    0: "Write one hint that points the student to the next idea or step.",
}

CLAIM_TYPES = """{"type":"determinant","value":q} {"type":"rank","value":r} {"type":"trace","value":q}
{"type":"eigenvalue","value":q} {"type":"eigenvector","eigenvalue":q,"vector":[...]}
{"type":"charpoly","coefficients":[...]} {"type":"invertible","value":true}
{"type":"row_op","start":[[...]],"op":"R2 <- R2 - 3R1","matrix":[[...]]} (matrix is the result of applying op to
  start; leave out start if op is applied to A itself)
{"type":"AtA","matrix":[[...]]} {"type":"Atb","vector":[...]}
{"type":"minor","row":1,"col":1,"value":q} {"type":"cofactor","row":1,"col":1,"value":q} (1-based)
{"type":"arithmetic","expr":"3*4-2*5","value":2}"""

HINT_RULES = f"""Never state the final answer, in words, numbers or claims.
Every number you state that is not in the problem must also appear in a claim. Claim types:
{CLAIM_TYPES}
A hint with no computed values has "claims": [].
{NUMBERS}"""


def solution_format(topic: str) -> str:
    return SOLUTION_FORMAT.format(answer=ANSWER_FORMAT[topic], steps=STEP_FORMAT[topic], op_rules=OP_RULES,
                                  numbers=NUMBERS)


def student_block(problem: dict) -> str:
    lines = ["Student's solution, step by step:"]
    for st in problem["student_steps"]:
        value = st.get("step_value")
        lines.append(f"Step {st['step_no']}: {st.get('step_text', '')}" + (f" => {value}" if value else ""))
    return "\n".join(lines)


def solver(problem: dict) -> str:
    return ("You are the Solver Agent of a linear algebra tutor. Solve the problem exactly, by hand-style "
            f"working.\n\nProblem:\n{problem['problem']}\n\n{solution_format(problem['topic'])}")


def solver_retry(problem: dict, previous: str, reason: str) -> str:
    return (f"{solver(problem)}\n\nYour previous attempt was:\n{previous}\n\n"
            f"An exact checker rejected it: {reason}\n"
            "Recompute carefully and send the full corrected solution in the same JSON format.")


def reask(prompt: str) -> str:
    return f"{prompt}\n\nYour last reply was not valid JSON. Reply with one JSON object only."


def single_llm(problem: dict) -> str:
    topic = problem["topic"]
    if problem.get("student_steps"):
        task = (f"Problem:\n{problem['problem']}\n\n{student_block(problem)}\n\n"
                "Find the first wrong step of the student's solution, or null if every step is right. "
                "Then write three graded hints for the student.\n")
        head = '{"first_wrong_step": k or null, "misconception": label or null, '
    else:
        # No hints on D1 or D2 in this paper (decision of 2026-11-08): solve only.
        return ("You are a linear algebra tutor.\n\n"
                f"Problem:\n{problem['problem']}\n\nSolve the problem.\n\n"
                "Work out the real values. Every field must hold computed numbers, never a placeholder or null.\n"
                "Reply with one JSON object only, no prose:\n"
                f'{{"steps": [...], "answer": {ANSWER_FORMAT[topic]}}}\n'
                f'Each entry of "steps" has this form: {STEP_FORMAT[topic]}.\n{OP_RULES}\n{NUMBERS}')
    levels = "\n".join(HINT_LEVELS[k] for k in (1, 2, 3))
    return ("You are a linear algebra tutor.\n\n" + task + f"{levels}\n{HINT_RULES}\n\n"
            "Work out the real values. Every field must hold computed numbers, never a placeholder or null.\n"
            "Reply with one JSON object only, no prose:\n"
            f'{head}"steps": [...], "answer": {ANSWER_FORMAT[topic]}, '
            '"hints": [{"level":1,"text":"...","claims":[...]},{"level":2,...},{"level":3,...}]}\n'
            f'Each entry of "steps" has this form: {STEP_FORMAT[topic]}.\n{OP_RULES}')


def self_check(prompt: str, previous: str) -> str:
    return (f"{prompt}\n\nHere is a reply you wrote earlier:\n{previous}\n\n"
            "Check every step and the final answer for mistakes. Then send the full reply again in the same JSON "
            'format, corrected if needed, with one extra field "self_check": "passed" if it was already right '
            'or "corrected" if you changed it.')


def diagnostic(problem: dict, taxonomy: list, reference: str | None, check_result: str | None) -> str:
    parts = [f"You are the Diagnostic Agent of a linear algebra tutor.\n\nProblem:\n{problem['problem']}",
             student_block(problem)]
    if reference:
        parts.append(f"A correct reference solution:\n{reference}")
    if check_result:
        parts.append(f"An exact checker found: {check_result}")
    if taxonomy:
        labels = "\n".join(f"- {t}" for t in taxonomy)
        parts.append(f"Choose exactly one misconception label from this list:\n{labels}")
        parts.append('Reply with one JSON object only: {"first_wrong_step": k or null, "misconception": "<label>"}')
    else:
        parts.append('Reply with one JSON object only: {"first_wrong_step": k or null}')
    return "\n\n".join(parts) + "\nSteps are numbered from 1."


def hint(problem: dict, level: int, answer_text: str | None, earlier: list, misconception: str | None,
         wrong_step: int | None, rejected: str | None = None) -> str:
    parts = [f"You are the Hint Agent of a linear algebra tutor.\n\nProblem:\n{problem['problem']}"]
    if problem.get("student_steps"):
        parts.append(student_block(problem))
        if wrong_step is not None:
            parts.append(f"The student's first wrong step is step {wrong_step}.")
    else:
        parts.append("The student is stuck and has not started.")
    if misconception:
        parts.append(f"Likely misconception: {misconception}")
    if answer_text:
        parts.append(f"Reference solution, for your eyes only. The student must not see its final answer:\n{answer_text}")
    if earlier:
        parts.append("Hints already given:\n" + "\n".join(f"- {h}" for h in earlier))
    parts.append(HINT_LEVELS[level])
    parts.append(HINT_RULES)
    if rejected:
        parts.append(f"Your previous hint was rejected because: {rejected}\nWrite a new hint that fixes this.")
    parts.append('Reply with one JSON object only: {"text": "...", "claims": [...]}')
    return "\n\n".join(parts)


def _answer_only(solution: dict) -> str:
    return compact({"steps": solution.get("steps"), "answer": solution.get("answer")})


def llm_verify_solution(problem: dict, solution: dict) -> str:
    return (f"You are the Verification Agent of a linear algebra tutor.\n\nProblem:\n{problem['problem']}\n\n"
            f"Proposed solution:\n{_answer_only(solution)}\n\n"
            "Judge whether the final answer and every step are mathematically correct.\n"
            'Reply with one JSON object only: {"verdict": "correct" or "incorrect", "reason": "short"}')


def llm_verify_hint(problem: dict, solution: dict, text: str, claims: list) -> str:
    return (f"You are the Verification Agent of a linear algebra tutor.\n\nProblem:\n{problem['problem']}\n\n"
            f"Reference solution:\n{_answer_only(solution)}\n\nHint for the student:\n{text}\n"
            f"Claims in the hint: {compact(claims)}\n\n"
            "The hint must be mathematically correct and must not give away the final answer.\n"
            'Reply with one JSON object only: {"verdict": "correct" or "incorrect", "reason": "short"}')


def detector(problem: dict) -> str:
    return (f"Problem:\n{problem['problem']}\n\n{student_block(problem)}\n\n"
            "Find the first step that contains a mathematical error. Steps are numbered from 1.\n"
            'Reply with one JSON object only: {"first_wrong_step": k, or null if every step is correct}')


def compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
