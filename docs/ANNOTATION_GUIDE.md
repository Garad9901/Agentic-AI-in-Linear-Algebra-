# D3 annotation guide

For the two annotators and anyone transcribing student solutions. Annotators are named in [ANNOTATORS.md](ANNOTATORS.md). Label independently, and do not discuss a solution until both of you have finished it.

Real student data stays in `data/private/d3.csv`, which is git-ignored. Remove names and roll numbers before transcribing. Never commit this file, and never send anything but anonymised transcriptions to an API.

## The one rule that matters most

**Write the row operation in `step_text`.** For example: `R2 <- R2 - 3R1`.

Without it, the step checker cannot catch arithmetic slips on full-rank matrices. Every invertible n × n matrix is row-equivalent to every other, so a wrong entry on an invertible matrix still looks valid. When the operation is written down, the checker applies it exactly and compares the result.

Take A = [[1, 2], [3, 4]], where the student writes [[1, 2], [0, -3]]. The correct second row is [0, -2].

| What you write in `step_text` | What the checker does |
|---|---|
| `R2 <- R2 - 3R1` | Reports wrong arithmetic at this step |
| *(nothing, or "eliminate")* | Passes the step, because both matrices are invertible |

Copy the operation the student wrote. If they did not write one, work out the operation they evidently meant and write that. If you cannot tell what they meant, leave it out and say so in the rest of `step_text`.

### Accepted forms

| Operation | Write |
|---|---|
| Add a multiple of a row | `R2 <- R2 - 3R1` |
| Scale a row | `R2 <- (1/2)R2` or `R2 <- R2/2` |
| Swap rows | `R1 <-> R2` |
| Combine rows | `R2 <- 2R2 - R1` |
| Several at once | `R2 <- R2 - 3R1, R3 <- R3 + R1` |

Put the operation alone on the first line of `step_text`. Anything else the student wrote goes on later lines. The checker reads only the first line, so `R2 <- R2 - 3R1 to clear the entry` would not be read as an operation.

## Columns

Use one row per step per annotator, as in `data/d3_template.csv`.

| Column | What goes in it |
|---|---|
| `solution_id` | `S017-P2` means student S017, problem 2. The part before the `-` keeps all of one student's work in the same split. |
| `topic` | One of `rref_rank`, `determinant`, `inverse`, `eigenpairs`, `diagonalisation`, `least_squares` |
| `problem` | The problem, including its matrix as `A = [[1,2],[3,4]]` (and `b = [5,6]` for least squares) |
| `step_no` | 1, 2, 3, ... One step is one line of written work. |
| `step_text` | The row operation alone on the first line, then anything else the student wrote |
| `step_value` | The matrix, vector or value of that step, in JSON: `[[1,2],[0,-2]]`, `[1,2]`, `-3/2`. Leave empty if the step has none. |
| `annotator` | Your name. Use `gold` only for the adjudicated labels agreed after discussion. |
| `step_label` | `correct` or `wrong` |
| `first_error_step` | The `step_no` of the first wrong step. Leave empty if the solution has no error. |
| `misconception` | One category from `data/taxonomy.json`. Leave empty if there is no error. |

## Transcribing

- Copy the student's numbers exactly, mistakes included. Never correct them.
- Write fractions as `"3/2"` strings inside JSON, for example `[[1,"3/2"],[0,1]]`. Do not convert them to decimals.
- For determinants by elimination, give each matrix in `step_value`, and give the final determinant as a value.
- For cofactor expansion, give each minor or term as a value. The checker only compares the last value with det(A), so a wrong minor is located at that last value, not at the minor. Your own `first_error_step` label should still name the step with the real mistake.
- For inverses, write the whole augmented matrix [A | I] in each step.

## Labelling

- **`step_label`:** mark a step `wrong` if it does not follow from the step before it. A step that correctly continues from an earlier mistake is `correct`, because only the first error is the one we locate.
- **`first_error_step`:** this is the first step marked `wrong`.
- **`misconception`:** the taxonomy is built by open coding on the dev split only, then frozen and tagged `taxonomy-v1` before the test split is annotated (PREREGISTRATION.md s4.4). Do not annotate test solutions before that tag exists.

## Agreement

We report Cohen's κ on your independent labels, from before any discussion. Disagreements are then settled by discussion and recorded in rows with `annotator` set to `gold`. Do not change your own rows after discussion.
