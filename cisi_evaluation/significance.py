"""
significance.py

Statistical tools for deciding whether a difference between two retrieval
models is real or just noise from having only 76 queries.

Why not just compare the averages?
    A model with MAP 0.22 vs another with MAP 0.21 looks "better", but a
    different random sample of queries could easily reverse that. Two tools
    answer this:

    * bootstrap_ci            how uncertain is ONE model's average score?
    * paired_permutation_test is the difference between TWO models bigger than
                              what chance alone would produce?

    Because every model is run on the same queries, the comparison is PAIRED
    (query by query), which is much more sensitive than comparing two
    independent averages.

    When many models are compared, some "significant" results appear by luck
    alone (at p < 0.05, about 1 in 20 comparisons of identical models would).
    holm_correction adjusts the p-values to account for this.
"""

import random


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=0):
    """
    Percentile bootstrap confidence interval for the mean of per-query scores.

    Resample the queries WITH replacement many times, take the mean of each
    resample, and report the middle (1 - alpha) of those means. Returns
    (mean, lower, upper).
    """
    values = list(values)
    rng = random.Random(seed)  # fixed seed -> identical numbers on every run
    n = len(values)

    resampled_means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(sample) / n)
    resampled_means.sort()

    lower = resampled_means[int((alpha / 2) * n_resamples)]
    upper = resampled_means[int((1 - alpha / 2) * n_resamples) - 1]
    return mean(values), lower, upper


def paired_permutation_test(scores_a, scores_b, n_permutations=20000, seed=0):
    """
    Two-sided paired sign-flip randomisation test (Smucker et al., 2007).

    Null hypothesis: the two systems are interchangeable, so for each query
    the sign of (score_a - score_b) is arbitrary. We randomly flip the signs
    many times and count how often the mean difference is at least as extreme
    as the one actually observed. Returns (mean_difference, p_value).
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(diffs)
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)

    extreme = 0
    for _ in range(n_permutations):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total / n) >= observed - 1e-15:
            extreme += 1

    # "+1" in both places is the standard correction that stops the estimated
    # p-value from ever being exactly 0.
    return sum(diffs) / n, (extreme + 1) / (n_permutations + 1)


def holm_correction(p_values):
    """
    Holm-Bonferroni adjustment for multiple comparisons.

    Sort the p-values from smallest to largest. The i-th smallest (0-based) is
    multiplied by (m - i), where m is the number of comparisons, and the
    results are made non-decreasing. Controls the chance of ANY false
    positive while being less conservative than plain Bonferroni.
    Returns the adjusted p-values in the original order.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])

    adjusted = [0.0] * m
    running_max = 0.0
    for position, original_index in enumerate(order):
        value = (m - position) * p_values[original_index]
        running_max = max(running_max, value)      # keep the sequence monotone
        adjusted[original_index] = min(1.0, running_max)
    return adjusted
