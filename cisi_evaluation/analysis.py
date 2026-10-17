"""
analysis.py

Deeper analysis that goes beyond a table of averages:

  * win_tie_loss         on how many queries is model A better than model B?
  * spearman             does performance depend on query length or on how
                         many relevant documents a query has?
  * best_and_worst       which individual queries does a model handle best/worst?
  * bm25_parameter_study does the "best" k1/b depend on WHICH metric you use?
  * lm_lambda_study      how does the smoothing weight affect the language model?
"""

import random

from metrics import evaluate_run
from runs import build_run
from bm25 import BM25
from lm_jelinek_mercer import LMJelinekMercer


# ----------------------------------------------------------------------
# Per-query comparison
# ----------------------------------------------------------------------
def win_tie_loss(scores_a, scores_b, tolerance=1e-9):
    """
    Counts queries where A beats B, where they tie, and where A loses.
    Averages can hide that one model wins on most queries but loses badly on
    a few (or the reverse), so this complements the mean.
    """
    wins = ties = losses = 0
    for a, b in zip(scores_a, scores_b):
        if abs(a - b) <= tolerance:
            ties += 1
        elif a > b:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


def _average_ranks(values):
    """Ranks 1..n of the values; tied values share the average of their ranks."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        # Extend j over every value tied with order[i].
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1  # +1 because ranks start at 1
        for position in range(i, j + 1):
            ranks[order[position]] = average_rank
        i = j + 1
    return ranks


def _pearson(a, b):
    """Pearson correlation of two equal-length lists (0.0 if either is constant)."""
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a == 0 or var_b == 0:
        return 0.0
    return cov / (var_a * var_b) ** 0.5


def spearman(x, y):
    """
    Spearman rank correlation in [-1, 1]: the Pearson correlation of the
    RANKS of x and y. Chosen over Pearson because it only assumes a monotone
    relationship and is not distorted by outliers. Returns 0.0 if either
    variable is constant (correlation is undefined).
    """
    return _pearson(_average_ranks(x), _average_ranks(y))


def spearman_permutation_p(x, y, n_permutations=10000, seed=0):
    """
    Two-sided p-value for a Spearman correlation, by permutation.

    If x and y were unrelated, any pairing of their values would be equally
    likely. Shuffle y many times and count how often the shuffled correlation
    is at least as strong as the observed one. With only 76 queries, a
    correlation of +0.1 is easily noise; this says which ones are not.
    """
    rx, ry = _average_ranks(x), _average_ranks(y)
    observed = abs(_pearson(rx, ry))
    rng = random.Random(seed)

    extreme = 0
    shuffled = list(ry)
    for _ in range(n_permutations):
        rng.shuffle(shuffled)  # shuffling the ranks == shuffling the values
        if abs(_pearson(rx, shuffled)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)


def query_features(queries, qrels, preprocessor, query_ids):
    """
    Two properties of each query that might explain why some are harder:
        n_terms     how many index terms the query has after preprocessing
        n_relevant  how many documents are judged relevant for it
    """
    return {
        qid: {
            "n_terms": len(preprocessor.process(queries[qid])),
            "n_relevant": len(qrels[qid]),
        }
        for qid in query_ids
    }


def split_by_length(features, query_ids):
    """
    Splits queries into a SHORT and a LONG half around the median number of
    index terms (queries at or below the median count as short). Returns
    (short_ids, long_ids, median_length).
    """
    lengths = sorted(features[q]["n_terms"] for q in query_ids)
    median = lengths[len(lengths) // 2]
    short = [q for q in query_ids if features[q]["n_terms"] <= median]
    long_ = [q for q in query_ids if features[q]["n_terms"] > median]
    return short, long_, median


def best_and_worst(per_query, metric, n=3):
    """Returns ([(qid, value) best n], [(qid, value) worst n]) for a metric."""
    ranked = sorted(per_query.items(), key=lambda item: item[1][metric], reverse=True)
    best = [(qid, row[metric]) for qid, row in ranked[:n]]
    worst = [(qid, row[metric]) for qid, row in ranked[-n:]]
    return best, worst


# ----------------------------------------------------------------------
# Parameter studies
# ----------------------------------------------------------------------
def bm25_parameter_study(index, preprocessor, queries, qrels, k1_values, b_values, metric_list):
    """
    Evaluates BM25 on every (k1, b) pair, keeping several metrics at once.
    Returns {metric: {(k1, b): mean_score}}.

    The point: if different metrics prefer different parameters, then "the
    best BM25 setting" is only meaningful relative to a chosen metric.
    """
    query_ids = sorted(qrels)
    grids = {metric: {} for metric in metric_list}

    for k1 in k1_values:
        for b in b_values:
            model = BM25(index, k1=k1, b=b, k3=8.0)
            run = build_run(model.search, preprocessor, queries, query_ids)
            means = evaluate_run(run, qrels)["mean"]
            for metric in metric_list:
                grids[metric][(k1, b)] = means[metric]
    return grids


def lm_lambda_study(index, preprocessor, queries, qrels, lambdas, metric_list):
    """
    Evaluates the Jelinek-Mercer language model for each smoothing weight.
    Returns {lambda: {metric: mean_score}}.
    """
    query_ids = sorted(qrels)
    results = {}
    for lam in lambdas:
        model = LMJelinekMercer(index, lam=lam)
        run = build_run(model.search, preprocessor, queries, query_ids)
        means = evaluate_run(run, qrels)["mean"]
        results[lam] = {metric: means[metric] for metric in metric_list}
    return results
