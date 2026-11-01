"""Statistics fixed by PREREGISTRATION.md section 6.

  McNemar (exact binomial when discordant pairs < 25, else chi-square with continuity correction)
  matched-pairs odds ratio b/c with exact 95% CI (Clopper-Pearson on b/(b+c))
  Holm-Bonferroni across the primary family
  Wilcoxon signed-rank with matched-pairs rank-biserial r
  cluster bootstrap 95% CI, 10,000 resamples (cluster = problem for D1, student for D3)
"""
from __future__ import annotations

import math
import random

from scipy import stats as st


def mcnemar(x, y):
    """x, y: paired 0/1 outcomes (1 = event, e.g. response wrong). Returns dict.
    b = x=1,y=0 ; c = x=0,y=1."""
    assert len(x) == len(y)
    b = sum(1 for a, d in zip(x, y) if a and not d)
    c = sum(1 for a, d in zip(x, y) if d and not a)
    n = b + c
    if n == 0:
        p, method = 1.0, "no discordant pairs"
    elif n < 25:
        p, method = st.binomtest(b, n, 0.5).pvalue, "exact binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n
        p, method = float(st.chi2.sf(chi2, 1)), "chi-square, continuity corrected"
    orr, lo, hi = matched_or(b, c)
    return {"b": b, "c": c, "n_pairs": len(x), "p": float(p), "method": method,
            "odds_ratio": orr, "or_ci_low": lo, "or_ci_high": hi}


def matched_or(b, c, alpha=0.05):
    """Matched-pairs OR = b / c with exact conditional CI."""
    n = b + c
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    lo_p = st.beta.ppf(alpha / 2, b, c + 1) if b > 0 else 0.0
    hi_p = st.beta.ppf(1 - alpha / 2, b + 1, c) if c > 0 else 1.0
    orr = b / c if c > 0 else float("inf")
    lo = lo_p / (1 - lo_p) if lo_p < 1 else float("inf")
    hi = hi_p / (1 - hi_p) if hi_p < 1 else float("inf")
    return orr, float(lo), float(hi)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def wilcoxon_rb(x, y):
    """Wilcoxon signed-rank on paired scores; returns p and matched-pairs rank-biserial r."""
    d = [a - b for a, b in zip(x, y) if a != b]
    if not d:
        return {"p": 1.0, "r_rb": 0.0, "n_nonzero": 0}
    ranks = st.rankdata([abs(v) for v in d])
    w_pos = sum(r for r, v in zip(ranks, d) if v > 0)
    w_neg = sum(r for r, v in zip(ranks, d) if v < 0)
    p = st.wilcoxon(x, y, zero_method="wilcox").pvalue
    return {"p": float(p), "r_rb": float((w_pos - w_neg) / (w_pos + w_neg)), "n_nonzero": len(d)}


def cluster_bootstrap(values, clusters, stat=None, B=10000, seed=20261024, alpha=0.05):
    """Percentile CI for stat(values) resampling whole clusters with replacement."""
    stat = stat or (lambda v: sum(v) / len(v) if v else float("nan"))
    groups = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    keys = list(groups)
    rng = random.Random(seed)
    est = []
    for _ in range(B):
        sample = []
        for _k in range(len(keys)):
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        est.append(stat(sample))
    est.sort()
    lo = est[int(math.floor(alpha / 2 * B))]
    hi = est[min(B - 1, int(math.ceil((1 - alpha / 2) * B)) - 1)]
    return stat(list(values)), lo, hi


def mean_sd(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def cohen_kappa(a, b, weights=None):
    """Cohen's kappa; weights=None (nominal) or 'quadratic' (ordinal ints)."""
    from collections import Counter
    cats = sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    if n == 0 or k < 2:
        return float("nan")
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    ra, rb = Counter(a), Counter(b)
    def w(i, j):
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2
        return 0.0 if i == j else 1.0
    po = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(w(i, j) * ra[cats[i]] * rb[cats[j]] for i in range(k) for j in range(k)) / (n * n)
    return 1 - po / pe if pe else float("nan")
