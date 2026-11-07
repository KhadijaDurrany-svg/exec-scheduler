"""Small, dependency-light statistics for honest benchmark reporting.

Only numpy is used. All randomness is seeded so results are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    seed: int = 0,
    n_boot: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return (mean, lo, hi)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    seed: int = 0,
    n_perm: int = 10_000,
) -> float:
    """Two-sided paired permutation p-value for mean(a) - mean(b).

    Signs of the per-pair differences are flipped at random; the p-value is the
    fraction of permutations whose |mean difference| >= the observed one.
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError("paired test requires equal-length samples")
    diff = arr_a - arr_b
    n = diff.size
    if n == 0:
        return 1.0
    observed = abs(float(diff.mean()))
    if np.allclose(diff, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = np.abs((signs * diff).mean(axis=1))
    # +1 in num/denom: never report p == 0 from a finite permutation sample
    count = int(np.count_nonzero(perm_means >= observed - 1e-12))
    return (count + 1) / (n_perm + 1)
