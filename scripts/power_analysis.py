"""Sample-size calculation for the primary student-data comparison (RQ2).

Paired design: the same erroneous student solutions are scored by the
verified system and by an LLM-only detector. Outcome per solution:
first incorrect step located exactly (1) or not (0). Test: McNemar.

Formula (Connor, 1987, Biometrics 43:207-211; reference to be opened and
checked in Phase 6) for McNemar sample size:
    n = [z_a * sqrt(p_disc) + z_b * sqrt(p_disc - d^2)]^2 / d^2
where p_disc = p10 + p01 (proportion of discordant pairs) and d = p10 - p01.

The assumed discordant proportions below are ASSUMPTIONS fixed before data
collection, not results. Change them only by logging a deviation.
"""
from math import sqrt, ceil
from scipy.stats import norm

P10 = 0.25   # verified system right, LLM-only wrong (assumed)
P01 = 0.05   # verified system wrong, LLM-only right (assumed)
POWER = 0.80
ALPHA_FAMILY = 0.05
N_PRIMARY_COMPARISONS = 4   # Holm-Bonferroni family; worst case uses alpha/4


def mcnemar_n(p10, p01, alpha, power):
    p_disc, d = p10 + p01, p10 - p01
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return ceil((z_a * sqrt(p_disc) + z_b * sqrt(p_disc - d * d)) ** 2 / d ** 2)


if __name__ == "__main__":
    n_uncorrected = mcnemar_n(P10, P01, ALPHA_FAMILY, POWER)
    n_holm_worst = mcnemar_n(P10, P01, ALPHA_FAMILY / N_PRIMARY_COMPARISONS, POWER)
    print(f"Erroneous solutions needed, alpha=0.05:            {n_uncorrected}")
    print(f"Erroneous solutions needed, alpha=0.05/{N_PRIMARY_COMPARISONS} (Holm worst case): {n_holm_worst}")
    print("Target in plan: >= 100 erroneous solutions in the TEST split.")
