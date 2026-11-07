"""One test group per verifier check. Each has a passing case and at least one failing case."""
import sympy as sp
import pytest

import grader
from data import gen_d1, gen_d2
from verifier import check_claim, check_steps, leak_check, parse_ops, verify, verify_hint
from verifier.exact import ParseError, evaluate, is_rref, num, to_json

A2 = {"A": [[1, 2], [3, 4]]}
SING = {"A": [[1, 2], [2, 4]]}
EIG = {"A": [[2, 1], [1, 2]]}  # eigenvalues 1, 3
REP = {"A": [[2, 0, 0], [0, 2, 0], [0, 0, 3]]}  # 2 is repeated
LSQ = {"A": [[1, 0], [0, 1], [1, 1]], "b": [1, 1, 0]}  # x = [1/3, 1/3]


# ---- exact parsing
@pytest.mark.parametrize("raw,want", [(3, 3), ("-7/4", sp.Rational(-7, 4)), ("0.25", sp.Rational(1, 4)),
                                      (0.5, sp.Rational(1, 2)), ("\\frac{1}{2}", sp.Rational(1, 2)),
                                      ("−3", -3)])
def test_num_accepts(raw, want):
    assert num(raw) == want


@pytest.mark.parametrize("raw", [True, "x", "1/0", None, "__import__('os')", float("nan")])
def test_num_rejects(raw):
    with pytest.raises(ParseError):
        num(raw)


def test_evaluate_is_exact_and_safe():
    assert evaluate("3*4-2*5") == 2
    assert evaluate("2(3-1)") == 4
    assert evaluate("1/3+1/6") == sp.Rational(1, 2)
    with pytest.raises(ParseError):
        evaluate("__import__('os').system('x')")


# ---- rref
def test_is_rref():
    assert is_rref(sp.Matrix([[1, 0, 2], [0, 1, 3], [0, 0, 0]]))
    assert not is_rref(sp.Matrix([[1, 1], [0, 1]]))  # entry above a pivot
    assert not is_rref(sp.Matrix([[2, 0], [0, 1]]))  # pivot is not 1
    assert not is_rref(sp.Matrix([[0, 0], [1, 0]]))  # zero row above a pivot


def test_rref_passes():
    assert verify("rref_rank", A2, {"rref": [[1, 0], [0, 1]], "rank": 2}).ok
    assert verify("rref_rank", SING, {"rref": [[1, 2], [0, 0]], "rank": 1}).ok


def test_rref_rejects_not_rref():
    v = verify("rref_rank", A2, {"rref": [[1, 2], [0, 1]], "rank": 2})
    assert not v.ok and "reduced row echelon" in v.reason


def test_rref_rejects_not_row_equivalent():
    v = verify("rref_rank", SING, {"rref": [[1, 3], [0, 0]], "rank": 1})
    assert not v.ok and "row-equivalent" in v.reason


def test_rank_must_equal_pivots():
    v = verify("rref_rank", SING, {"rref": [[1, 2], [0, 0]], "rank": 2})
    assert not v.ok and "pivots" in v.reason


# ---- determinant
def test_determinant():
    assert verify("determinant", A2, {"determinant": -2}).ok
    assert verify("determinant", A2, {"determinant": "-4/2"}).ok
    assert not verify("determinant", A2, {"determinant": 2}).ok


# ---- inverse
def test_inverse():
    assert verify("inverse", A2, {"inverse": [[-2, 1], ["3/2", "-1/2"]]}).ok
    assert not verify("inverse", A2, {"inverse": [[-2, 1], ["3/2", "1/2"]]}).ok
    assert not verify("inverse", A2, {"inverse": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}).ok


# ---- eigenpairs
PAIRS = [{"eigenvalue": 1, "eigenvector": [1, -1]}, {"eigenvalue": 3, "eigenvector": [1, 1]}]


def test_eigenpairs_pass():
    assert verify("eigenpairs", EIG, {"eigenpairs": PAIRS}).ok
    scaled = [{"eigenvalue": 1, "eigenvector": ["1/2", "-1/2"]}, {"eigenvalue": 3, "eigenvector": [5, 5]}]
    assert verify("eigenpairs", EIG, {"eigenpairs": scaled}).ok


def test_eigenpairs_reject_bad_vector():
    bad = [{"eigenvalue": 1, "eigenvector": [1, 1]}, PAIRS[1]]
    assert "A*v" in verify("eigenpairs", EIG, {"eigenpairs": bad}).reason


def test_eigenpairs_reject_zero_vector():
    bad = [{"eigenvalue": 1, "eigenvector": [0, 0]}, PAIRS[1]]
    assert "zero vector" in verify("eigenpairs", EIG, {"eigenpairs": bad}).reason


def test_eigenpairs_reject_wrong_multiset():
    v = verify("eigenpairs", EIG, {"eigenpairs": [PAIRS[1], PAIRS[1]]})
    assert not v.ok
    v = verify("eigenpairs", EIG, {"eigenpairs": [PAIRS[1]]})
    assert "multiset" in v.reason


def test_eigenpairs_repeated_value_needs_independent_vectors():
    good = [{"eigenvalue": 2, "eigenvector": [1, 0, 0]}, {"eigenvalue": 2, "eigenvector": [0, 1, 0]},
            {"eigenvalue": 3, "eigenvector": [0, 0, 1]}]
    assert verify("eigenpairs", REP, {"eigenpairs": good}).ok
    dup = [good[0], good[0], good[2]]
    assert "dependent" in verify("eigenpairs", REP, {"eigenpairs": dup}).reason


# ---- diagonalisation
def test_diagonalisation_pass():
    assert verify("diagonalisation", EIG, {"P": [[1, 1], [-1, 1]], "D": [[1, 0], [0, 3]]}).ok


def test_diagonalisation_rejects_singular_p():
    v = verify("diagonalisation", EIG, {"P": [[1, 1], [1, 1]], "D": [[3, 0], [0, 3]]})
    assert "singular" in v.reason


def test_diagonalisation_rejects_wrong_product():
    v = verify("diagonalisation", EIG, {"P": [[1, 1], [-1, 1]], "D": [[3, 0], [0, 1]]})
    assert "P*D" in v.reason


def test_diagonalisation_rejects_non_diagonal_d():
    v = verify("diagonalisation", EIG, {"P": [[1, 0], [0, 1]], "D": [[2, 1], [1, 2]]})
    assert "not diagonal" in v.reason


# ---- least squares
def test_least_squares():
    assert verify("least_squares", LSQ, {"x": ["1/3", "1/3"]}).ok
    assert not verify("least_squares", LSQ, {"x": [1, 1]}).ok
    assert not verify("least_squares", LSQ, {"x": [1, 1, 1]}).ok


# ---- malformed input never raises
@pytest.mark.parametrize("topic", gen_d1.TOPICS)
@pytest.mark.parametrize("candidate", [None, {}, {"rref": "x", "determinant": "x", "inverse": 3, "eigenpairs": [1],
                                                  "P": [[1]], "D": 1, "x": "y", "rank": []}])
def test_malformed_answer_fails_cleanly(topic, candidate):
    problem = {"A": [[1, 2], [3, 4]], "b": [1, 2]}
    assert not verify(topic, problem, candidate).ok


# ---- row operation parser
@pytest.mark.parametrize("text", ["R2 <- R2 - 3R1", "R2 -> R2 - 3*R1", "R2 = R2 - 3 R1", "R_2 ← R_2 − 3R_1",
                                  "R2 - 3R1 -> R2"])
def test_parse_ops_forms(text):
    ops = parse_ops(text)
    assert ops is not None and ops[0][1] == 1


def test_parse_ops_swap_scale_and_multiple():
    assert parse_ops("R1 <-> R2") == [("swap", 0, 1)]
    assert parse_ops("swap R1 and R3") == [("swap", 0, 2)]
    assert parse_ops("R2 <- (1/2)R2")[0][2][0][1] == sp.Rational(1, 2)
    assert parse_ops("R2 <- R2/3")[0][2][0][1] == sp.Rational(1, 3)
    assert len(parse_ops("R2 <- R2 - R1, R3 <- R3 + 2R1")) == 2
    assert parse_ops("multiply through") is None


# ---- step checker
def test_steps_all_valid():
    steps = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]},
             {"op": "R2 <- (-1/2)R2", "matrix": [[1, 2], [0, 1]]},
             {"op": "R1 <- R1 - 2R2", "matrix": [[1, 0], [0, 1]]}, 2]
    assert check_steps("rref_rank", A2, steps).first_bad is None


def test_steps_not_row_equivalent():
    r = check_steps("rref_rank", SING, [[[1, 2], [0, 1]]])
    assert r.first_bad == 0 and "row-equivalent" in r.reason


def test_steps_wrong_arithmetic_with_stated_op():
    # Row-equivalent to A because both are invertible, so only the stated operation exposes it.
    steps = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}]
    r = check_steps("rref_rank", A2, steps)
    assert r.first_bad == 0 and "wrong arithmetic" in r.reason


def test_steps_op_is_read_from_text():
    steps = [{"text": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]}]
    assert check_steps("rref_rank", A2, steps).first_bad == 0


def test_steps_weak_is_reported_not_hidden():
    r = check_steps("rref_rank", A2, [[[1, 2], [0, -3]]])
    assert r.first_bad is None and r.weak == [0]


def test_steps_finds_the_first_of_two_errors():
    steps = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]},
             {"op": "R1 <- R1 + R2", "matrix": [[1, 1], [0, -2]]},
             {"op": "R2 <- R2 + R1", "matrix": [[9, 9], [9, 9]]}]
    assert check_steps("rref_rank", A2, steps).first_bad == 1


def test_steps_later_steps_follow_from_the_wrong_one_not_from_a():
    steps = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]},
             {"op": "R2 <- (-1/3)R2", "matrix": [[1, 2], [0, 1]]}]
    assert check_steps("rref_rank", A2, steps).first_bad == 0


def test_steps_wrong_rank_value():
    assert check_steps("rref_rank", SING, [[[1, 2], [0, 0]], 2]).first_bad == 1


def test_steps_determinant_value_and_arithmetic():
    ok = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}, {"expr": "1*(-2)", "value": -2}]
    assert check_steps("determinant", A2, ok).first_bad is None
    wrong = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}, {"value": 2}]
    assert check_steps("determinant", A2, wrong).first_bad == 1
    arithmetic = [{"expr": "3*4-2*5", "value": 3}]
    r = check_steps("determinant", A2, arithmetic)
    assert r.first_bad == 0 and "arithmetic" in r.reason


def test_steps_determinant_factor():
    assert check_steps("determinant", A2, [{"matrix": [[1, 2], [0, 1]], "factor": -2}]).first_bad is None
    assert check_steps("determinant", A2, [{"matrix": [[1, 2], [0, 1]], "factor": 2}]).first_bad == 0


def test_steps_inverse_augmented_and_final():
    n = [{"op": "R2 <- R2 - 3R1", "matrix": [[1, 2, 1, 0], [0, -2, -3, 1]]},
         {"op": "R2 <- (-1/2)R2", "matrix": [[1, 2, 1, 0], [0, 1, "3/2", "-1/2"]]},
         {"op": "R1 <- R1 - 2R2", "matrix": [[1, 0, -2, 1], [0, 1, "3/2", "-1/2"]]},
         {"matrix": [[-2, 1], ["3/2", "-1/2"]]}]
    assert check_steps("inverse", A2, n).first_bad is None
    # The right block goes wrong: only an augmented row-equivalence or op check can see it.
    bad = [{"matrix": [[1, 2, 1, 0], [0, -2, -3, 2]]}]
    assert check_steps("inverse", A2, bad).first_bad == 0
    assert check_steps("inverse", A2, [{"matrix": [[1, 0], [0, 1]]}]).first_bad == 0


def test_steps_eigen():
    ok = [{"coefficients": [1, -4, 3]}, {"eigenvalue": 3}, {"eigenvalue": 3, "vector": [1, 1]},
          {"eigenvalue": 1, "matrix": [[1, 1], [0, 0]]}]
    assert check_steps("eigenpairs", EIG, ok).first_bad is None
    assert check_steps("eigenpairs", EIG, [{"coefficients": [1, -4, 4]}]).first_bad == 0
    assert check_steps("eigenpairs", EIG, [{"eigenvalue": 2}]).first_bad == 0
    assert check_steps("diagonalisation", EIG, [{"eigenvalue": 3, "vector": [1, -1]}]).first_bad == 0
    assert check_steps("eigenpairs", EIG, [{"eigenvalue": 1, "matrix": [[1, 2], [0, 0]]}]).first_bad == 0


def test_steps_least_squares():
    AtA, Atb = [[2, 1], [1, 2]], [1, 1]
    ok = [{"matrix": AtA}, {"vector": Atb}, {"matrix": [[2, 1, 1], [1, 2, 1]]},
          {"op": "R2 <- 2R2 - R1", "matrix": [[2, 1, 1], [0, 3, 1]]}, {"vector": ["1/3", "1/3"]}]
    assert check_steps("least_squares", LSQ, ok).first_bad is None
    assert check_steps("least_squares", LSQ, [{"matrix": [[2, 1], [1, 3]]}]).first_bad == 0
    assert check_steps("least_squares", LSQ, [{"vector": [5, 5]}]).first_bad == 0
    assert check_steps("least_squares", LSQ, [{"matrix": [[2, 1, 2], [1, 2, 1]]}]).first_bad == 0


def test_steps_text_only_is_skipped_not_wrong():
    r = check_steps("determinant", A2, [{"text": "expand along the first row"}, -2])
    assert r.first_bad is None and r.unchecked == [0]


def test_steps_empty():
    assert check_steps("rref_rank", A2, []).first_bad is None


# ---- leak checker
def test_leak_determinant_forms():
    answer = {"determinant": -7}
    assert leak_check("The determinant is -7.", "determinant", answer).leaked
    assert leak_check("so det(A) = −7", "determinant", answer).leaked
    assert leak_check("det A equals minus 7", "determinant", answer).leaked
    assert not leak_check("Expand along the first row.", "determinant", answer).leaked


def test_leak_small_scalar_needs_context():
    answer = {"determinant": 3}
    assert not leak_check("Add 3 times row 1 to row 3.", "determinant", answer).leaked
    assert leak_check("The answer is 3", "determinant", answer).leaked


def test_leak_large_or_fractional_scalar_counts_anywhere():
    assert leak_check("You should reach 142.", "determinant", {"determinant": 142}).leaked
    assert leak_check(r"You should reach \frac{-1}{2}.", "determinant", {"determinant": "-1/2"}).leaked


def test_leak_matrix():
    answer = {"inverse": [[-2, 1], ["3/2", "-1/2"]]}
    assert leak_check("The inverse is [[-2, 1], [3/2, -1/2]]", "inverse", answer).leaked
    assert leak_check(r"\begin{pmatrix} -2 & 1 \\ \frac{3}{2} & \frac{-1}{2} \end{pmatrix}", "inverse", answer).leaked
    assert not leak_check("Start from [A | I] and clear the first column.", "inverse", answer).leaked


def test_leak_rank_and_rref():
    answer = {"rref": [[1, 2], [0, 0]], "rank": 1}
    assert leak_check("the rank is 1", "rref_rank", answer).leaked
    assert leak_check("you get [[1, 2], [0, 0]]", "rref_rank", answer).leaked
    assert not leak_check("use row 1 to clear row 2", "rref_rank", answer).leaked
    full = {"rref": [[1, 0], [0, 1]], "rank": 2}
    assert leak_check("A reduces to the identity matrix", "rref_rank", full).leaked


def test_leak_eigenvalues_need_all_of_them():
    answer = {"eigenpairs": PAIRS}
    assert leak_check("The eigenvalues are 1 and 3.", "eigenpairs", answer).leaked
    assert not leak_check("One eigenvalue is 3.", "eigenpairs", answer).leaked
    diag = {"P": [[1, 1], [-1, 1]], "D": [[1, 0], [0, 3]]}
    assert leak_check("The eigenvalues are 1 and 3.", "diagonalisation", diag).leaked


def test_leak_eigenvalue_forms():
    answer = {"eigenpairs": [{"eigenvalue": 0, "eigenvector": [1, 1]}, {"eigenvalue": -1, "eigenvector": [1, 2]}]}
    assert leak_check("The eigenvalues are 0 and ‑1.", "eigenpairs", answer).leaked
    assert leak_check(r"so \lambda_1 = 0 and \lambda_2 = -1", "eigenpairs", answer).leaked
    claims = [{"type": "eigenvalue", "value": 0}, {"type": "eigenvalue", "value": -1}]
    assert leak_check("Compute it.", "eigenpairs", answer, claims).leaked


def test_leak_eigenvalues_ignore_incidental_numbers():
    answer = {"eigenpairs": [{"eigenvalue": 0, "eigenvector": [1, 1]}, {"eigenvalue": -1, "eigenvector": [1, 2]}]}
    text = "Form A-λI = [[1-λ, -1],[2, -2-λ]]. Set the polynomial to 0 to get the eigenvalues."
    assert not leak_check(text, "eigenpairs", answer).leaked
    text = "Recall that the eigenvalues are the roots of the characteristic polynomial, with (A-λI)v = 0 and -1."
    assert not leak_check(text, "eigenpairs", answer).leaked


def test_every_hyphen_form_is_a_minus():
    for ch in "‐‑‒–−":
        assert num(f"{ch}2") == -2
        assert evaluate(f"5{ch}2") == 3
    assert leak_check("The determinant is ‑7.", "determinant", {"determinant": -7}).leaked
    assert not verify_hint(A2, "You get ‑3 here.", []).verified  # 3 is in A, but -3 is not
    assert parse_ops("R2 ← R2 ‑ 3R1")[0][2][1][1] == -3


def test_leak_least_squares_vector():
    answer = {"x": ["1/3", "1/3"]}
    assert leak_check("so x = [1/3, 1/3]", "least_squares", answer).leaked
    assert not leak_check("form the normal equations first", "least_squares", answer).leaked


def test_leak_through_claims():
    assert leak_check("Compute it.", "determinant", {"determinant": -7},
                      [{"type": "determinant", "value": -7}]).leaked


def test_leak_row_labels_are_not_numbers():
    answer = {"x": [2, 3]}
    assert not leak_check("Use row 2 and row 3", "least_squares", answer).leaked


# ---- hint claims
@pytest.mark.parametrize("claim", [
    {"type": "determinant", "value": -2}, {"type": "rank", "value": 2}, {"type": "trace", "value": 5},
    {"type": "row_op", "op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]},
    {"type": "row_op", "op": "R2 <- R2 - 3R1", "matrix": [[1, 2, 1, 0], [0, -2, -3, 1]]},
    {"type": "minor", "row": 1, "col": 1, "value": 4}, {"type": "cofactor", "row": 1, "col": 2, "value": -3},
    {"type": "arithmetic", "expr": "3*4-2*5", "value": 2}, {"type": "invertible", "value": True},
    {"type": "charpoly", "coefficients": [1, -5, -2]},
])
def test_claim_certified(claim):
    assert check_claim(A2, claim) == "certified"


@pytest.mark.parametrize("claim", [
    {"type": "determinant", "value": 2}, {"type": "rank", "value": 1}, {"type": "trace", "value": 4},
    {"type": "row_op", "op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -3]]},
    {"type": "cofactor", "row": 1, "col": 2, "value": 3}, {"type": "arithmetic", "expr": "3*4", "value": 13},
    {"type": "invertible", "value": False},
])
def test_claim_refuted(claim):
    assert check_claim(A2, claim) == "refuted"


@pytest.mark.parametrize("claim", ["text", {"type": "vibes"}, {"type": "determinant"}])
def test_claim_uncheckable(claim):
    assert check_claim(A2, claim) == "uncheckable"


def test_claim_eigen_and_normal_equations():
    assert check_claim(EIG, {"type": "eigenvalue", "value": 3}) == "certified"
    assert check_claim(EIG, {"type": "eigenvalue", "value": 2}) == "refuted"
    assert check_claim(EIG, {"type": "eigenvector", "eigenvalue": 3, "vector": [1, 1]}) == "certified"
    assert check_claim(LSQ, {"type": "AtA", "matrix": [[2, 1], [1, 2]]}) == "certified"
    assert check_claim(LSQ, {"type": "Atb", "vector": [1, 1]}) == "certified"
    assert check_claim(LSQ, {"type": "Atb", "vector": [1, 2]}) == "refuted"


def test_row_op_from_a_later_matrix_must_stay_connected_to_a():
    swapped = [[3, 4], [1, 2]]
    good = {"type": "row_op", "start": swapped, "op": "R1 <- (1/3)R1", "matrix": [["1", "4/3"], [1, 2]]}
    assert check_claim(A2, good) == "certified"
    wrong_result = {**good, "matrix": [[1, 2], [1, 2]]}
    assert check_claim(A2, wrong_result) == "refuted"
    singular = {"A": [[1, 2], [2, 4]]}
    unconnected = {"type": "row_op", "start": [[1, 0], [0, 1]], "op": "R1 <-> R2", "matrix": [[0, 1], [1, 0]]}
    assert check_claim(singular, unconnected) == "refuted"


def test_hint_verified_needs_every_claim_and_every_number():
    claim = {"type": "row_op", "op": "R2 <- R2 - 3R1", "matrix": [[1, 2], [0, -2]]}
    assert verify_hint(A2, "Apply R2 <- R2 - 3R1 to get -2 in position (2,2).", [claim]).verified
    assert verify_hint(A2, "What happens to the first column?", []).verified
    assert not verify_hint(A2, "You should get 77 here.", []).verified
    assert not verify_hint(A2, "The determinant is 5.", [{"type": "determinant", "value": 5}]).verified
    assert not verify_hint(A2, "No numbers.", [{"type": "vibes"}]).verified


# ---- D1 and grader
@pytest.fixture(scope="module")
def d1_sample():
    return gen_d1.generate(n_per_cell=3)


def test_d1_is_deterministic_and_has_valid_entries(d1_sample):
    assert len(d1_sample) == 6 * 3 * 3
    for p in d1_sample:
        assert max(abs(e) for row in p["A"] for e in row) <= 9
        assert all(isinstance(e, int) for row in p["A"] for e in row)
    assert gen_d1.generate(n_per_cell=1)[0] == {**d1_sample[0], "split": gen_d1.generate(n_per_cell=1)[0]["split"]}


def test_d1_truth_passes_the_verifier_and_the_grader(d1_sample):
    for p in d1_sample:
        answer = grader.answer_of(p)
        assert verify(p["topic"], p, answer).ok, p["id"]
        assert grader.grade(p, answer), p["id"]


def _perturb(x):
    if isinstance(x, list):
        return [_perturb(x[0]), *x[1:]]
    return to_json(num(x) + 1)


def test_verifier_and_grader_agree_on_corrupted_answers(d1_sample):
    for p in d1_sample:
        answer = dict(grader.answer_of(p))
        key = {"rref_rank": "rref", "determinant": "determinant", "inverse": "inverse", "diagonalisation": "P",
               "least_squares": "x"}.get(p["topic"])
        if key:
            answer[key] = _perturb(answer[key])
        else:
            first = dict(answer["eigenpairs"][0])
            first["eigenvector"] = _perturb(first["eigenvector"])
            answer["eigenpairs"] = [first, *answer["eigenpairs"][1:]]
        assert verify(p["topic"], p, answer).ok == grader.grade(p, answer), p["id"]


def test_grader_accepts_other_valid_eigenvectors():
    p = {"topic": "eigenpairs", "A": EIG["A"], "ground_truth": {
        "eigenvalues": [{"value": 1, "multiplicity": 1}, {"value": 3, "multiplicity": 1}],
        "eigenspaces": [{"eigenvalue": 1, "basis": [[-1, 1]]}, {"eigenvalue": 3, "basis": [[1, 1]]}]}}
    assert grader.grade(p, {"eigenpairs": PAIRS})
    assert not grader.grade(p, {"eigenpairs": [PAIRS[1]]})


def test_follows_from_working_compares_with_the_last_value():
    follows = grader.follows_from_working
    assert follows("determinant", [{"value": 5}], {"determinant": 5})
    assert not follows("determinant", [{"value": 4}], {"determinant": 5})
    assert not follows("determinant", [{"value": 5}, {"value": 4}], {"determinant": 5})  # not the last value
    assert follows("inverse", [{"matrix": [[1, 0, 2, 3], [0, 1, 4, 5]]}], {"inverse": [[2, 3], [4, 5]]})
    assert follows("least_squares", [{"vector": [1, 2]}], {"x": [1, 2]})
    assert not follows("least_squares", None, {"x": [1, 2]})
    assert not follows("least_squares", [], {"x": [1, 2]})
    assert follows("rref_rank", [{"matrix": [[1, 0], [0, 1]]}, {"value": 2}], {"rref": [[1, 0], [0, 1]], "rank": 2})
    pair = {"eigenpairs": [{"eigenvalue": 3, "eigenvector": [1, 1]}]}
    assert follows("eigenpairs", [{"eigenvalue": 3, "vector": [1, 1]}], pair)
    assert not follows("eigenpairs", [{"eigenvalue": 2, "vector": [1, 1]}], pair)  # same vector, other eigenvalue
    assert not follows("eigenpairs", [{"text": "x"}], {"eigenpairs": []})


def test_d1_matches_prereg_v1():
    problems = gen_d1.load()
    assert len(problems) == 270 and gen_d1.SIZES == [3, 4, 5]
    assert sum(p["split"] == "dev" for p in problems) == 54
    for topic in gen_d1.TOPICS:
        for size in gen_d1.SIZES:
            cell = [p for p in problems if p["topic"] == topic and p["size"] == size]
            assert len(cell) == 15 and sum(p["split"] == "dev" for p in cell) == 3


def test_d1_stored_truth_passes_verifier_and_grader():
    for p in gen_d1.load():
        answer = grader.answer_of(p)
        assert verify(p["topic"], p, answer).ok and grader.grade(p, answer), p["id"]


def test_d2_runnable_truth_passes_verifier_and_grader():
    problems = gen_d2.load()
    runnable = [p for p in problems if p["runnable"]]
    assert len(problems) == 660 and len(runnable) == 210
    for p in runnable:
        answer = grader.answer_of(p)
        assert verify(p["topic"], p, answer).ok and grader.grade(p, answer), p["id"]


def test_parse_matrix_reads_latex():
    bs = chr(92)
    latex = f"A = {bs}begin{{bmatrix}} -3 & 5 {bs}{bs} 2 & 0 {bs}end{{bmatrix}}"
    assert gen_d2.parse_matrix(latex) == [[-3, 5], [2, 0]]
