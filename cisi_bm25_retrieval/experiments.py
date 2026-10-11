"""
experiments.py

Builds the retrieval system and runs the experiments used in the report:

  1. build_system()         - load CISI, preprocess, build the inverted index
  2. preprocessing_ablation - does stemming / stop word removal help?
  3. parameter_sweep        - effect of BM25's k1 and b on MAP
  4. heldout_tuning         - honest estimate of tuned performance
                              (tune on half the queries, test on the other half)
"""

from bm25 import BM25
from dataset_loader import load_cisi
from evaluation import evaluate
from inverted_index import InvertedIndex
from preprocessing import Preprocessor
from tfidf_baseline import TfidfCosine


def build_index(documents, preprocessor):
    """
    Indexes title + abstract of every document. The author field is
    deliberately excluded: author names rarely appear in the queries and
    would only add noise terms to the vocabulary.
    """
    index = InvertedIndex()
    for doc in documents:
        terms = preprocessor.process(doc.title + " " + doc.text)
        index.add_document(doc.doc_id, terms)
    index.finalize()
    return index


def build_system(data_folder, use_stopwords=True, use_stemming=True):
    documents, queries, qrels = load_cisi(data_folder)
    preprocessor = Preprocessor(use_stopwords=use_stopwords, use_stemming=use_stemming)
    index = build_index(documents, preprocessor)
    return documents, queries, qrels, preprocessor, index


def preprocessing_ablation(documents, queries, qrels, k1=1.2, b=0.75):
    """Evaluates BM25 under all four stop word / stemming combinations."""
    rows = []
    for use_stop in (False, True):
        for use_stem in (False, True):
            pre = Preprocessor(use_stopwords=use_stop, use_stemming=use_stem)
            index = build_index(documents, pre)
            bm25 = BM25(index, k1=k1, b=b)
            result = evaluate(bm25.search, pre, queries, qrels)
            rows.append({
                "stopwords": use_stop,
                "stemming": use_stem,
                "vocab": len(index.postings),
                "MAP": result["MAP"],
                "P@10": result["P@10"],
                "R@10": result["R@10"],
            })
    return rows


def parameter_sweep(index, preprocessor, queries, qrels, k1_values, b_values):
    """Returns {(k1, b): MAP} for every combination."""
    grid = {}
    for k1 in k1_values:
        for b in b_values:
            bm25 = BM25(index, k1=k1, b=b)
            grid[(k1, b)] = evaluate(bm25.search, preprocessor, queries, qrels)["MAP"]
    return grid


def heldout_tuning(index, preprocessor, queries, qrels, k1_values, b_values):
    """
    Tuning k1/b on the very queries used to report results would overstate
    performance. So: split the judged queries into two halves (alternating
    query ids), pick the best (k1, b) on the TUNING half, then measure that
    choice - and the default (1.2, 0.75) - on the unseen TEST half.
    """
    ids = sorted(qrels)
    tune_qrels = {q: qrels[q] for i, q in enumerate(ids) if i % 2 == 0}
    test_qrels = {q: qrels[q] for i, q in enumerate(ids) if i % 2 == 1}

    grid = parameter_sweep(index, preprocessor, queries, tune_qrels, k1_values, b_values)
    best_k1, best_b = max(grid, key=grid.get)

    tuned = evaluate(BM25(index, best_k1, best_b).search, preprocessor, queries, test_qrels)
    default = evaluate(BM25(index, 1.2, 0.75).search, preprocessor, queries, test_qrels)
    tfidf = evaluate(TfidfCosine(index).search, preprocessor, queries, test_qrels)

    return {
        "n_tune": len(tune_qrels),
        "n_test": len(test_qrels),
        "best_k1": best_k1,
        "best_b": best_b,
        "tune_map": grid[(best_k1, best_b)],
        "test_map_tuned": tuned["MAP"],
        "test_map_default": default["MAP"],
        "test_map_tfidf": tfidf["MAP"],
    }
