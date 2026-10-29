"""Small, dependency-free statistics for the brain tournaments.

* wilson_ci      - 95% CI for a proportion (valid for small n and p near 0/1;
                   the naive mean +- 1.96*sd/sqrt(n) is NOT).
* mean_ci_t      - 95% CI for a mean using Student's t and the *sample* sd.
* mann_whitney_u - two-sided Mann-Whitney U test (normal approximation with
                   tie and continuity correction) for two samples.
* fisher_exact   - exact two-sided Fisher test for two win/loss counts.
"""
import math

# two-sided 97.5% Student-t critical values, df = 1..30
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval (lo, hi) for k successes out of n trials."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci_t(xs):
    """(mean, half_width) of a 95% t-interval; half_width 0 for n < 2."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    m = sum(xs) / n
    if n < 2:
        return (m, 0.0)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    t = _T975[n - 2] if n - 1 <= 30 else 1.96
    return (m, t * sd / math.sqrt(n))


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U. Returns (U_a, p_value)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    tie_term = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu = n1 * n2 / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return (u1, 1.0)
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return (u1, min(1.0, p))


def fisher_exact(k1, n1, k2, n2):
    """Exact two-sided Fisher p-value for k1/n1 vs k2/n2 successes."""
    K, N = k1 + k2, n1 + n2

    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)

    lo, hi = max(0, K - n2), min(n1, K)
    p_obs = pmf(k1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))
