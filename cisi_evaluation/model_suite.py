"""
model_suite.py

Defines the retrieval models that are evaluated and produces one run (the
stored rankings) for each. Keeping this in one place guarantees that main.py
and verify_with_trec_eval.py evaluate exactly the same rankings.

Models
------
Random        reference floor: documents in random order (seeded)
TF-IDF        vector space model, cosine similarity (earlier assignment)
BM25-set      BM25 with the query treated as a plain SET of terms (k3 off)
BM25          Okapi BM25, k1=1.2, b=0.75, k3=8 - the project's final model
LM-JM 0.1     language model, Jelinek-Mercer smoothing, lambda = 0.1
LM-JM 0.7     language model, Jelinek-Mercer smoothing, lambda = 0.7

The two lambda values were fixed IN ADVANCE from the published guidance of
Zhai & Lafferty (2001): small lambda suits short keyword queries, larger lambda
suits long verbose queries. CISI queries are verbose, so 0.7 is the setting
that guidance predicts to be better. Neither value was chosen by looking at
the scores; a full lambda sweep is reported separately.
"""

from bm25 import BM25
from lm_jelinek_mercer import LMJelinekMercer
from runs import build_run, random_run
from tfidf_baseline import TfidfCosine

# Display order for every table and chart.
MODEL_ORDER = ["Random", "TF-IDF", "BM25-set", "BM25", "LM-JM 0.1", "LM-JM 0.7"]

MODEL_DESCRIPTIONS = {
    "Random": "random document order (reference floor)",
    "TF-IDF": "TF-IDF weights + cosine similarity",
    "BM25-set": "BM25 k1=1.2 b=0.75, query treated as a set (no query tf)",
    "BM25": "BM25 k1=1.2 b=0.75 k3=8 (final model of assignment 6)",
    "LM-JM 0.1": "language model, Jelinek-Mercer, lambda=0.1",
    "LM-JM 0.7": "language model, Jelinek-Mercer, lambda=0.7",
}


def build_all_runs(documents, queries, qrels, preprocessor, index):
    """Returns {model_name: run} for every model, over the judged queries."""
    query_ids = sorted(qrels)

    search_functions = {
        "TF-IDF": TfidfCosine(index).search,
        "BM25-set": BM25(index, k1=1.2, b=0.75, k3=None).search,
        "BM25": BM25(index, k1=1.2, b=0.75, k3=8.0).search,
        "LM-JM 0.1": LMJelinekMercer(index, lam=0.1).search,
        "LM-JM 0.7": LMJelinekMercer(index, lam=0.7).search,
    }

    runs = {"Random": random_run([d.doc_id for d in documents], query_ids, seed=0)}
    for name, search_fn in search_functions.items():
        runs[name] = build_run(search_fn, preprocessor, queries, query_ids)
    return runs
