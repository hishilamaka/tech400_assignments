"""
evaluation.py

Standard ranked-retrieval metrics, computed against the CISI relevance
judgments (CISI.REL). Relevance is binary: a document is either listed as
relevant for a query or it is not.

Metrics
-------
Precision@k   fraction of the top-k results that are relevant
Recall@k      fraction of all relevant documents found in the top-k
AP            average precision: mean of precision@r taken at the rank r of
              each relevant document (relevant documents never retrieved
              contribute 0)
MAP           mean of AP over all evaluated queries - the headline metric
MRR           mean reciprocal rank of the first relevant result
11-pt curve   interpolated precision at recall 0.0, 0.1, ... 1.0

Significance
------------
paired_permutation_test compares two systems query by query. A MAP
difference on 76 queries can easily be noise, so differences are only
called real when this test says so.
"""

import random


def precision_at_k(ranked_ids, relevant, k):
    top = ranked_ids[:k]
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def average_precision(ranked_ids, relevant):
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def reciprocal_rank(ranked_ids, relevant):
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def interpolated_precision_curve(ranked_ids, relevant):
    """
    11-point interpolated precision: for each recall level r in
    {0.0, 0.1, ..., 1.0}, the highest precision achieved at any point where
    recall >= r.
    """
    if not relevant:
        return [0.0] * 11

    points = []  # (recall, precision) after each relevant document is found
    hits = 0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            points.append((hits / len(relevant), hits / rank))

    curve = []
    for level in [i / 10 for i in range(11)]:
        candidates = [p for r, p in points if r >= level - 1e-12]
        curve.append(max(candidates) if candidates else 0.0)
    return curve


def evaluate(search_fn, preprocessor, queries, qrels, k=10):
    """
    Runs every judged query through the retrieval system and computes metrics.

    search_fn     function(list_of_terms) -> ranked list of (doc_id, score)
    preprocessor  the same Preprocessor used to build the index
    queries       {query_id: query_text}
    qrels         {query_id: set(relevant_doc_ids)}

    Only queries that have relevance judgments are evaluated (76 of the 112
    CISI queries). Returns a dict with per-query rows and macro averages.
    """
    per_query = []
    curves = []

    for qid in sorted(qrels):
        if qid not in queries:
            continue
        relevant = qrels[qid]
        terms = preprocessor.process(queries[qid])
        ranked_ids = [doc_id for doc_id, _ in search_fn(terms)]

        per_query.append({
            "query_id": qid,
            "n_relevant": len(relevant),
            f"P@{k}": precision_at_k(ranked_ids, relevant, k),
            f"R@{k}": recall_at_k(ranked_ids, relevant, k),
            "AP": average_precision(ranked_ids, relevant),
            "RR": reciprocal_rank(ranked_ids, relevant),
        })
        curves.append(interpolated_precision_curve(ranked_ids, relevant))

    n = len(per_query)
    mean = lambda key: sum(row[key] for row in per_query) / n

    return {
        "k": k,
        "n_queries": n,
        "per_query": per_query,
        f"P@{k}": mean(f"P@{k}"),
        f"R@{k}": mean(f"R@{k}"),
        "MAP": mean("AP"),
        "MRR": mean("RR"),
        "pr_curve": [sum(c[i] for c in curves) / n for i in range(11)],
    }


def paired_permutation_test(scores_a, scores_b, n_permutations=20000, seed=0):
    """
    Two-sided paired sign-flip randomisation test on per-query scores
    (e.g. the AP of system A vs system B on the same queries).

    Under the null hypothesis the two systems are interchangeable, so the
    sign of each per-query difference is arbitrary. We randomly flip signs
    and count how often the mean difference is at least as extreme as the
    observed one. Returns (mean_difference, p_value).
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed = abs(sum(diffs) / len(diffs))
    rng = random.Random(seed)

    extreme = 0
    for _ in range(n_permutations):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total / len(diffs)) >= observed - 1e-15:
            extreme += 1

    mean_diff = sum(diffs) / len(diffs)
    return mean_diff, (extreme + 1) / (n_permutations + 1)
