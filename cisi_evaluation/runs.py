"""
runs.py

A "run" is the ranked list of documents a retrieval model returned for every
query:

    run = { query_id: [(doc_id, score), (doc_id, score), ...] }   best first

Separating "produce the rankings" from "measure the rankings" is standard IR
practice (it is how TREC evaluations work). It means every metric is computed
from the same stored rankings, and the rankings can be saved in TREC format and
checked with an independent tool (see verify_with_trec_eval.py).
"""

import os
import random


def build_run(search_fn, preprocessor, queries, query_ids):
    """
    Runs every query through a model.

    search_fn     function(list_of_terms) -> ranked [(doc_id, score), ...]
    preprocessor  the SAME Preprocessor used to build the index
    """
    run = {}
    for qid in query_ids:
        terms = preprocessor.process(queries[qid])
        run[qid] = search_fn(terms)
    return run


def random_run(all_doc_ids, query_ids, seed=0):
    """
    A reference "model" that ranks all documents in random order.

    It sets the floor of the scale: with no intelligence at all, what do P@10,
    MAP and nDCG look like? Any real model must beat this by a wide margin for
    its scores to mean anything. Seeded, so results are reproducible.
    """
    rng = random.Random(seed)
    run = {}
    for qid in query_ids:
        order = list(all_doc_ids)
        rng.shuffle(order)
        # Descending dummy scores so the list has the same shape as real runs.
        run[qid] = [(doc_id, float(len(order) - i)) for i, doc_id in enumerate(order)]
    return run


def oracle_run(qrels, query_ids, all_doc_ids):
    """
    A perfect ranking: for every query, all relevant documents first, then the
    rest. It is not a real model - it cheats by reading the answer key - but
    its scores are the CEILING of each metric. This matters on CISI, where a
    query can have 100+ relevant documents: Recall@10 can never exceed
    10/R, so even a flawless system scores well below 1 on it. Comparing a
    real model to the ceiling is fairer than comparing it to 1.0.
    """
    run = {}
    for qid in query_ids:
        relevant = sorted(qrels[qid])
        others = [d for d in all_doc_ids if d not in qrels[qid]]
        order = relevant + others
        run[qid] = [(doc_id, float(len(order) - i)) for i, doc_id in enumerate(order)]
    return run


def ranked_ids(run, qid):
    """Just the document ids of one query's ranking, best first."""
    return [doc_id for doc_id, _ in run[qid]]


def write_trec_run(run, path, tag):
    """
    Saves a run in the standard TREC format, one line per retrieved document:

        <query_id> Q0 <doc_id> <rank> <score> <run_tag>
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(run):
            for rank, (doc_id, score) in enumerate(run[qid], start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def write_trec_qrels(qrels, path):
    """Saves relevance judgments in TREC format:  <query_id> 0 <doc_id> <relevance>"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid]):
                f.write(f"{qid} 0 {doc_id} 1\n")
