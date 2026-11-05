"""
Comparisons between groups of training runs, with runs as the replicates. Each returns the two means, their
difference, a 95% bootstrap interval for it, a two-sided p-value, and whether the difference is "clear": the interval
excludes zero and p < 0.05.
"""

import functools
import itertools
import math

import numpy as np

NUM_RESAMPLES = 20_000  # bootstrap resamples, and permutations when there are too many to list exactly


@functools.lru_cache(maxsize=16)
def _splits(n: int, k: int) -> np.ndarray:
    """Every way to choose k of n runs, as rows of indices (cached: the same group sizes come up again and again)."""
    return np.array(list(itertools.combinations(range(n), k)), dtype=np.int16).reshape(-1, k)


def compare(a, b, rng: np.random.Generator) -> dict:
    """Two independent groups of runs, b minus a. The p-value comes from shuffling runs between the groups."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    difference = b.mean() - a.mean()
    bootstrap = rng.choice(b, (NUM_RESAMPLES, len(b))).mean(axis=1) - rng.choice(a, (NUM_RESAMPLES, len(a))).mean(axis=1)
    pooled = np.concatenate([a, b])
    if math.comb(len(pooled), len(b)) <= 200_000:  # few enough to try every split of runs between the groups
        splits = _splits(len(pooled), len(b))
    else:
        splits = np.argsort(rng.random((NUM_RESAMPLES, len(pooled))), axis=1)[:, : len(b)]
    in_b = pooled[splits].sum(axis=1)  # all splits at once
    shuffled = in_b / len(b) - (pooled.sum() - in_b) / len(a)
    as_extreme = np.abs(shuffled) >= abs(difference) - 1e-12
    return _result(a.mean(), b.mean(), bootstrap, float(as_extreme.mean()))


def compare_paired(a, b, rng: np.random.Generator) -> dict:
    """The same runs measured twice, b minus a per run. The p-value comes from flipping the sign of each difference."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    differences = b - a
    bootstrap = rng.choice(differences, (NUM_RESAMPLES, len(differences))).mean(axis=1)
    if len(differences) <= 16:  # few enough to try every combination of signs
        signs = np.array(list(itertools.product([1, -1], repeat=len(differences))))
    else:
        signs = rng.choice([1, -1], size=(NUM_RESAMPLES, len(differences)))
    flipped = np.abs((signs * differences).mean(axis=1))
    return _result(a.mean(), b.mean(), bootstrap, float(np.mean(flipped >= abs(differences.mean()) - 1e-12)))


def _result(mean_a: float, mean_b: float, bootstrap: list[float], p_value: float) -> dict:
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "a": float(mean_a),
        "b": float(mean_b),
        "difference": float(mean_b - mean_a),
        "ci95": [float(low), float(high)],
        "p_value": float(p_value),
        "clear": bool((low > 0 or high < 0) and p_value < 0.05),
    }
