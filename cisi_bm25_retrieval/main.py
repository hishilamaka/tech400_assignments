"""
main.py

End-to-end run of the CISI information retrieval system:

    1. Dataset overview
    2. Preprocessing demonstration
    3. Inverted index statistics
    4. Example queries with ranked results
    5. Evaluation: BM25 vs. TF-IDF baseline
    6. Preprocessing ablation (stop words / stemming)
    7. BM25 parameter study (k1, b) + held-out check

Usage:
    python main.py

Figures and a per-query CSV are written to the results/ folder.
"""

import csv
import os
import sys

from bm25 import BM25
from experiments import (build_system, heldout_tuning, parameter_sweep,
                         preprocessing_ablation)
from evaluation import evaluate, paired_permutation_test
from plots import plot_param_heatmap, plot_pr_curves
from tfidf_baseline import TfidfCosine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DEMO_QUERY_IDS = [1, 3, 14]   # a mixed case, a good case, and a failure case
K1_VALUES = [0.5, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0]
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]


def heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def shorten(text, width):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def show_dataset_overview(documents, queries, qrels):
    heading("1. DATASET OVERVIEW (CISI)")
    n_pairs = sum(len(v) for v in qrels.values())
    per_query = sorted(len(v) for v in qrels.values())
    print(f"Documents                    : {len(documents)}")
    print(f"Queries                      : {len(queries)}")
    print(f"Queries with judgments       : {len(qrels)}  (only these are evaluated)")
    print(f"Relevant (query, doc) pairs  : {n_pairs}")
    print(f"Relevant docs per query      : min {per_query[0]}, "
          f"median {per_query[len(per_query) // 2]}, max {per_query[-1]}")

    doc = documents[0]
    print(f"\nSample document [{doc.doc_id}]")
    print(f"  Title   : {doc.title}")
    print(f"  Author  : {doc.author}")
    print(f"  Abstract: {shorten(doc.text, 200)}")
    print(f"\nSample query [1]: {shorten(queries[1], 200)}")


def show_preprocessing(preprocessor, queries):
    heading("2. PREPROCESSING PIPELINE (example on query 1)")
    stages = preprocessor.trace(queries[1])
    print(f"Raw text      : {shorten(queries[1], 90)}")
    print(f"Tokenized     : {shorten(' '.join(stages['tokens']), 90)}")
    print(f"- stop words  : {shorten(' '.join(stages['no_stopwords']), 90)}")
    print(f"+ stemming    : {shorten(' '.join(stages['stemmed']), 90)}")


def show_index_stats(index):
    heading("3. INVERTED INDEX STATISTICS")
    for key, value in index.stats().items():
        print(f"{key.replace('_', ' '):<20}: {value}")
    print("\nTerms occurring in the most documents (term, document frequency):")
    print("  " + ", ".join(f"{t}({df})" for t, df in index.most_frequent_terms(10)))

    print("\nExample postings (term -> {doc_id: term frequency}):")
    for term in ["retriev", "librari"]:
        postings = dict(sorted(index.get_postings(term).items())[:6])
        print(f"  {term:<10} df={index.df(term):<4} {postings} ...")


def show_example_queries(bm25, preprocessor, documents, queries, qrels):
    heading("4. EXAMPLE QUERIES - BM25 TOP 5   ([REL] = judged relevant)")
    titles = {d.doc_id: d.title for d in documents}
    for qid in DEMO_QUERY_IDS:
        terms = preprocessor.process(queries[qid])
        results = bm25.search(terms, top_k=5)
        relevant = qrels.get(qid, set())
        print(f"\nQuery {qid}: {shorten(queries[qid], 100)}")
        print(f"  index terms: {' '.join(terms)}")
        for rank, (doc_id, score) in enumerate(results, start=1):
            mark = "[REL]" if doc_id in relevant else "     "
            print(f"  {rank}. {mark} doc {doc_id:<5} score {score:6.3f}  {shorten(titles[doc_id], 48)}")
        hits = sum(1 for d, _ in results if d in relevant)
        print(f"  -> {hits}/5 of the top results are relevant "
              f"({len(relevant)} relevant docs exist for this query)")

        full_ranking = [d for d, _ in bm25.search(terms)]
        first = next((i for i, d in enumerate(full_ranking, 1) if d in relevant), None)
        unreachable = sum(1 for d in relevant if d not in full_ranking)
        print(f"  -> first relevant document is at rank {first}"
              + (f"; {unreachable} relevant doc(s) share NO term with the query" if unreachable else ""))
        absent = sorted({t for t in terms if bm25.index.df(t) == 0})
        if absent:
            print(f"  -> query term(s) found nowhere in the collection: {', '.join(absent)}")


def print_metrics_table(results):
    print(f"\n{'Model':<32}{'P@10':>8}{'R@10':>8}{'MAP':>8}{'MRR':>8}")
    print("-" * 64)
    for name, r in results.items():
        print(f"{name:<32}{r['P@10']:>8.4f}{r['R@10']:>8.4f}{r['MAP']:>8.4f}{r['MRR']:>8.4f}")


def save_per_query_csv(result, path):
    rows = result["per_query"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Make sure output survives Windows consoles with a legacy code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    documents, queries, qrels, preprocessor, index = build_system(DATA_DIR)

    show_dataset_overview(documents, queries, qrels)
    show_preprocessing(preprocessor, queries)
    show_index_stats(index)

    bm25 = BM25(index, k1=1.2, b=0.75)
    show_example_queries(bm25, preprocessor, documents, queries, qrels)

    # ---- 5. Evaluation ------------------------------------------------
    heading("5. EVALUATION ON 76 JUDGED QUERIES")
    final_name = "BM25 (k1=1.2, b=0.75, k3=8)"
    nodup_name = "BM25, query as a set (no qtf)"
    base_name = "TF-IDF cosine (baseline)"
    results = {
        base_name: evaluate(TfidfCosine(index).search, preprocessor, queries, qrels),
        nodup_name: evaluate(BM25(index, 1.2, 0.75, k3=None).search, preprocessor, queries, qrels),
        final_name: evaluate(bm25.search, preprocessor, queries, qrels),
    }
    print_metrics_table(results)

    ap = lambda name: [row["AP"] for row in results[name]["per_query"]]
    print("\nPaired permutation tests on per-query average precision:")
    for a, b in [(final_name, nodup_name), (final_name, base_name)]:
        diff, p = paired_permutation_test(ap(a), ap(b))
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"  {a} vs {b}\n      mean AP difference {diff:+.4f}, p = {p:.4f}  ({verdict} at 0.05)")

    bm25_result = results[final_name]

    plot_pr_curves(
        {name: r["pr_curve"] for name, r in results.items()},
        os.path.join(RESULTS_DIR, "pr_curve.png"),
    )
    save_per_query_csv(bm25_result, os.path.join(RESULTS_DIR, "bm25_per_query.csv"))

    # ---- 6. Preprocessing ablation ------------------------------------
    heading("6. PREPROCESSING ABLATION (BM25, default parameters)")
    print(f"{'Stop words':<12}{'Stemming':<10}{'Vocab':>8}{'P@10':>8}{'R@10':>8}{'MAP':>8}")
    print("-" * 54)
    for row in preprocessing_ablation(documents, queries, qrels):
        print(f"{'removed' if row['stopwords'] else 'kept':<12}"
              f"{'on' if row['stemming'] else 'off':<10}{row['vocab']:>8}"
              f"{row['P@10']:>8.4f}{row['R@10']:>8.4f}{row['MAP']:>8.4f}")

    # ---- 7. Parameter study -------------------------------------------
    heading("7. BM25 PARAMETER STUDY (MAP)")
    grid = parameter_sweep(index, preprocessor, queries, qrels, K1_VALUES, B_VALUES)
    print(f"{'k1 \\ b':<8}" + "".join(f"{b:>8}" for b in B_VALUES))
    for k1 in K1_VALUES:
        print(f"{k1:<8}" + "".join(f"{grid[(k1, b)]:>8.4f}" for b in B_VALUES))
    best = max(grid, key=grid.get)
    print(f"\nBest on all 76 queries: k1={best[0]}, b={best[1]}  (MAP {grid[best]:.4f})")
    plot_param_heatmap(grid, K1_VALUES, B_VALUES, os.path.join(RESULTS_DIR, "param_heatmap.png"))

    held = heldout_tuning(index, preprocessor, queries, qrels, K1_VALUES, B_VALUES)
    print(f"\nHeld-out check (tune on {held['n_tune']} queries, test on {held['n_test']} unseen queries):")
    print(f"  best on tuning half : k1={held['best_k1']}, b={held['best_b']}  (MAP {held['tune_map']:.4f})")
    print(f"  test MAP, tuned     : {held['test_map_tuned']:.4f}")
    print(f"  test MAP, default   : {held['test_map_default']:.4f}")
    print(f"  test MAP, TF-IDF    : {held['test_map_tfidf']:.4f}   (baseline on the same unseen queries)")

    print(f"\nFigures and CSV saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
