from __future__ import annotations

from slotfit.stats import bootstrap_ci, paired_permutation_test


def test_bootstrap_ci_brackets_mean():
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean, lo, hi = bootstrap_ci(values, seed=1)
    assert abs(mean - 0.6) < 1e-9
    assert lo <= mean <= hi


def test_bootstrap_ci_single_value_is_degenerate():
    assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_deterministic():
    v = [0.1, 0.3, 0.9, 0.2, 0.7]
    assert bootstrap_ci(v, seed=3) == bootstrap_ci(v, seed=3)


def test_permutation_identical_samples_p_is_one():
    a = [0.5, 0.6, 0.7]
    assert paired_permutation_test(a, a, seed=0) == 1.0


def test_permutation_large_difference_small_p():
    a = [0.9, 0.95, 0.92, 0.88, 0.91, 0.93]
    b = [0.1, 0.15, 0.12, 0.08, 0.11, 0.13]
    p = paired_permutation_test(a, b, seed=0, n_perm=2000)
    assert 0.0 < p < 0.05


def test_permutation_p_in_unit_interval():
    a = [0.5, 0.4, 0.6, 0.55]
    b = [0.52, 0.39, 0.58, 0.54]
    p = paired_permutation_test(a, b, seed=2)
    assert 0.0 < p <= 1.0


def test_permutation_never_returns_zero():
    a = [1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = paired_permutation_test(a, b, seed=0, n_perm=500)
    assert p > 0.0
