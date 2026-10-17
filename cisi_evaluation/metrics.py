"""
metrics.py

Retrieval-evaluation metrics, implemented from their definitions.

Every function takes
    ranked    the document ids a model returned for ONE query, best first
    relevant  the SET of document ids judged relevant for that query

CISI relevance judgments are BINARY (a document is relevant or it is not), so
the "gain" of a relevant document is 1 and of any other document 0. Documents
without a judgment are treated as non-relevant, the usual convention.

Metrics, grouped by what they answer
------------------------------------
Set-based (does the top-k contain the right documents? order inside top-k is ignored)
    Precision@k   of the k documents returned, what fraction is relevant?
    Recall@k      of ALL relevant documents, what fraction is in the top k?
    F1@k          harmonic mean of Precision@k and Recall@k
    Success@k     is there at least one relevant document in the top k? (0/1)

Rank-aware (does the model put relevant documents EARLY?)
    AP            average precision: precision measured at the rank of each
                  relevant document, averaged over all relevant documents.
                  The mean over queries is MAP.
    RR            reciprocal rank of the first relevant document (mean = MRR)
    R-Precision   precision at rank R, where R = number of relevant documents
    nDCG@k        normalised discounted cumulative gain: relevant documents
                  are worth less the lower they appear, normalised by the best
                  possible ranking so scores lie in [0, 1]

Whole-ranking
    11-point interpolated precision-recall curve
"""

import math


# ----------------------------------------------------------------------
# Set-based metrics at a cutoff k
# ----------------------------------------------------------------------
def count_relevant_in_top_k(ranked, relevant, k):
    """Number of relevant documents among the first k results."""
    return sum(1 for doc_id in ranked[:k] if doc_id in relevant)


def precision_at_k(ranked, relevant, k):
    # The denominator is always k, even if fewer than k documents were
    # returned: a system that returns 3 results does not get credit for
    # "100% precision" on the 7 slots it left empty.
    return count_relevant_in_top_k(ranked, relevant, k) / k


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return count_relevant_in_top_k(ranked, relevant, k) / len(relevant)


def f1_at_k(ranked, relevant, k):
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    # Harmonic mean; defined as 0 when both are 0 to avoid dividing by zero.
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def success_at_k(ranked, relevant, k):
    return 1.0 if count_relevant_in_top_k(ranked, relevant, k) > 0 else 0.0


# ----------------------------------------------------------------------
# Rank-aware metrics
# ----------------------------------------------------------------------
def average_precision(ranked, relevant):
    """
    AP = (1 / R) * sum over relevant documents d of  Precision@rank(d)

    R is the TOTAL number of relevant documents for the query, so relevant
    documents the model never retrieved contribute 0 and pull AP down.
    """
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / rank  # precision at this relevant document
    return precision_sum / len(relevant)


def reciprocal_rank(ranked, relevant):
    """1 / (rank of the first relevant document), or 0 if none is retrieved."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def r_precision(ranked, relevant):
    """Precision at rank R, where R is the number of relevant documents."""
    if not relevant:
        return 0.0
    r = len(relevant)
    return count_relevant_in_top_k(ranked, relevant, r) / r


def dcg_at_k(ranked, relevant, k):
    """
    Discounted cumulative gain:   DCG@k = sum over ranks i=1..k of  gain_i / log2(i + 1)

    With binary relevance gain_i is 1 for a relevant document, else 0. The
    logarithmic discount makes a relevant document at rank 1 worth 1.0, at
    rank 3 worth 0.5, at rank 10 worth about 0.29 - so finding relevant
    documents early matters most.
    """
    total = 0.0
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            total += 1.0 / math.log2(rank + 1)
    return total


def ideal_dcg_at_k(num_relevant, k):
    """
    DCG of the best conceivable ranking: all relevant documents first. Only
    min(k, R) positions can hold a relevant document.
    """
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, num_relevant) + 1))


def ndcg_at_k(ranked, relevant, k):
    """nDCG@k = DCG@k / ideal DCG@k, so a perfect ranking scores exactly 1."""
    if not relevant:
        return 0.0
    ideal = ideal_dcg_at_k(len(relevant), k)
    return dcg_at_k(ranked, relevant, k) / ideal if ideal > 0 else 0.0


# ----------------------------------------------------------------------
# Whole-ranking curves
# ----------------------------------------------------------------------
def interpolated_precision_curve(ranked, relevant):
    """
    11-point interpolated precision-recall curve. At each recall level
    r in {0.0, 0.1, ..., 1.0} the value is the HIGHEST precision achieved at
    any point where recall >= r (the standard interpolation rule).
    """
    if not relevant:
        return [0.0] * 11

    points = []  # (recall, precision) each time a relevant document is found
    hits = 0
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            points.append((hits / len(relevant), hits / rank))

    curve = []
    for level in [i / 10 for i in range(11)]:
        candidates = [precision for recall, precision in points if recall >= level - 1e-12]
        curve.append(max(candidates) if candidates else 0.0)
    return curve


def metrics_at_every_k(ranked, relevant, k_max):
    """
    Precision@k, Recall@k and nDCG@k for EVERY k = 1..k_max in one pass.
    Used to draw how the metrics change as the user reads further down the
    ranking. Returns three lists, each of length k_max (index 0 is k=1).
    """
    precision, recall, ndcg = [], [], []
    hits = 0
    dcg = 0.0
    for k in range(1, k_max + 1):
        if k <= len(ranked) and ranked[k - 1] in relevant:
            hits += 1
            dcg += 1.0 / math.log2(k + 1)  # discount depends on this rank
        precision.append(hits / k)
        recall.append(hits / len(relevant) if relevant else 0.0)
        ideal = ideal_dcg_at_k(len(relevant), k)
        ndcg.append(dcg / ideal if ideal > 0 else 0.0)
    return precision, recall, ndcg


# ----------------------------------------------------------------------
# Evaluating a whole run
# ----------------------------------------------------------------------
P_CUTOFFS = (1, 3, 5, 10, 20)
R_CUTOFFS = (5, 10, 20, 50, 100)
F1_CUTOFFS = (10, 20)
NDCG_CUTOFFS = (5, 10, 20)
SUCCESS_CUTOFFS = (1, 5, 10)
CURVE_K_MAX = 100


def metric_names():
    """All per-query metric names, in display order."""
    names = [f"P@{k}" for k in P_CUTOFFS]
    names += [f"R@{k}" for k in R_CUTOFFS]
    names += [f"F1@{k}" for k in F1_CUTOFFS]
    names += [f"nDCG@{k}" for k in NDCG_CUTOFFS]
    names += [f"Success@{k}" for k in SUCCESS_CUTOFFS]
    names += ["AP", "RR", "RPrec"]
    return names


def evaluate_query(ranked, relevant):
    """Every metric for one query, as {metric_name: value}."""
    row = {}
    for k in P_CUTOFFS:
        row[f"P@{k}"] = precision_at_k(ranked, relevant, k)
    for k in R_CUTOFFS:
        row[f"R@{k}"] = recall_at_k(ranked, relevant, k)
    for k in F1_CUTOFFS:
        row[f"F1@{k}"] = f1_at_k(ranked, relevant, k)
    for k in NDCG_CUTOFFS:
        row[f"nDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
    for k in SUCCESS_CUTOFFS:
        row[f"Success@{k}"] = success_at_k(ranked, relevant, k)
    row["AP"] = average_precision(ranked, relevant)
    row["RR"] = reciprocal_rank(ranked, relevant)
    row["RPrec"] = r_precision(ranked, relevant)
    return row


def evaluate_run(run, qrels):
    """
    Evaluates a run (see runs.py) against the relevance judgments.

    Only queries that HAVE judgments are scored (76 of CISI's 112): a query
    with no judgments cannot be evaluated, since every result would look
    non-relevant.

    Returns a dict with
        per_query   {query_id: {metric: value}}
        mean        {metric: average over queries}   (AP -> MAP, RR -> MRR)
        curves      mean precision / recall / nDCG at every k = 1..100
        pr_curve    mean 11-point interpolated precision
        n_queries   number of queries evaluated
    """
    per_query = {}
    pr_curves = []
    sum_p = [0.0] * CURVE_K_MAX
    sum_r = [0.0] * CURVE_K_MAX
    sum_n = [0.0] * CURVE_K_MAX

    query_ids = sorted(qid for qid in qrels if qid in run)
    for qid in query_ids:
        relevant = qrels[qid]
        ranked = [doc_id for doc_id, _ in run[qid]]

        per_query[qid] = evaluate_query(ranked, relevant)
        pr_curves.append(interpolated_precision_curve(ranked, relevant))

        p, r, n = metrics_at_every_k(ranked, relevant, CURVE_K_MAX)
        for i in range(CURVE_K_MAX):
            sum_p[i] += p[i]
            sum_r[i] += r[i]
            sum_n[i] += n[i]

    n_queries = len(query_ids)
    mean = {
        name: sum(per_query[qid][name] for qid in query_ids) / n_queries
        for name in metric_names()
    }
    return {
        "per_query": per_query,
        "mean": mean,
        "curves": {
            "precision": [v / n_queries for v in sum_p],
            "recall": [v / n_queries for v in sum_r],
            "ndcg": [v / n_queries for v in sum_n],
        },
        "pr_curve": [sum(c[i] for c in pr_curves) / n_queries for i in range(11)],
        "n_queries": n_queries,
    }
