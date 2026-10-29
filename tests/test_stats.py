import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.stats import wilson_ci, mean_ci_t, mann_whitney_u, fisher_exact


def test_wilson_known_value():
    lo, hi = wilson_ci(9, 10)
    assert abs(lo - 0.596) < 0.01 and abs(hi - 0.982) < 0.01
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(0, 10)[1] > 0.2


def test_mean_ci_t():
    m, h = mean_ci_t([1, 2, 3, 4, 5])
    assert m == 3 and abs(h - 2.776 * (2.5 ** 0.5) / (5 ** 0.5)) < 1e-9


def test_mann_whitney_separated_and_equal():
    _, p_sep = mann_whitney_u(list(range(10)), list(range(20, 30)))
    _, p_eq = mann_whitney_u([1, 2, 3, 4], [1, 2, 3, 4])
    assert p_sep < 0.001 and p_eq > 0.9


def test_fisher_exact_known():
    assert abs(fisher_exact(3, 3, 0, 3) - 0.1) < 1e-9
    assert fisher_exact(5, 10, 5, 10) == 1.0
