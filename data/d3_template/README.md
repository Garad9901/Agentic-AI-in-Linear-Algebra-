# D3 transcription format (real student solutions)

**Superseded on 6 Nov 2026.** Use `data/d3_template.csv` and `docs/ANNOTATION_GUIDE.md`. This page describes the earlier JSONL format.

Real student data never goes in this repository. Keep it in `data/private/d3/` (git-ignored):

```
data/private/d3/problems.jsonl    the set problems students solved (same fields as D1: id, topic, size, question, A, b, solution)
data/private/d3/solutions.jsonl   one anonymised, transcribed solution per line
```

`python scripts/check_d3.py data/private/d3` validates both files and assigns the dev/test split by student
(seed 20261024, 30/70, PREREGISTRATION.md s4.3).

## One line of `solutions.jsonl`

```json
{
  "solution_id": "S017-P2",
  "student_id": "S017",
  "problem_id": "SET-inverse-3-a",
  "excluded": false,
  "exclusion_reason": null,
  "steps": [
    {"text": "Write [A | I]",
     "claim": {"type": "intermediate_matrix", "context": "augmented_A_I",
               "matrix": [[2,1,0,1,0,0],[1,3,1,0,1,0],[0,1,2,0,0,1]]}},
    {"text": "R2 <- 2R2 - R1",
     "claim": {"type": "intermediate_matrix", "context": "augmented_A_I",
               "matrix": [[2,1,0,1,0,0],[0,5,1,-1,2,0],[0,1,2,0,0,1]]}},
    {"text": "So the inverse is ...",
     "claim": {"type": "inverse", "matrix": [["3/5","-1/4","1/8"],["-1/4","1/2","-1/4"],["1/8","-1/4","5/8"]]}},
    {"text": "student wrote: 'since det is non-zero the matrix is invertible'"}
  ],
  "gold": {
    "annotator1": {"first_error_step": 2, "misconception": "ROW_OP_ARITHMETIC"},
    "annotator2": {"first_error_step": 2, "misconception": "ROW_OP_ARITHMETIC"},
    "first_error_step": 2,
    "misconception": "ROW_OP_ARITHMETIC"
  }
}
```

Rules for transcribers
* One step = one line of written work. Copy the student's numbers exactly, mistakes included.
* Give each step a `claim` in the claim schema (see `latutor/verifier.py`) when it states a matrix or value.
  A step with only words gets `text` and no `claim`; the verifier treats it as uncheckable.
* Row-reduction steps: `intermediate_matrix` with the right `context`
  (`A`, `augmented_A_I`, `A_minus_lambda_I` + `lambda`, `AtA`, `normal_augmented`).
* Determinant steps: `det_relation` (det A = c * det M) or `det_combination` (cofactor expansion).
* Step numbers in `gold` are 1-based. `first_error_step: null` means the solution has no error.
* `annotator1` / `annotator2` are the independent, pre-discussion labels (they give kappa);
  the top-level `first_error_step` / `misconception` are the agreed labels after discussion.
* No names, roll numbers or handwriting images in these files.
