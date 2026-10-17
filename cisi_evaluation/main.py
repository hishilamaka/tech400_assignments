"""
main.py

Complete evaluation of the CISI retrieval models.

    python main.py

Sections printed (each is meant to be screenshotted for the report):
    1. Evaluation set-up                 6. Statistical significance
    2. Headline metrics                  7. Per-query analysis
    3. Metrics at several cutoffs        8. Parameter sensitivity by metric
    4. Metric ceilings                   9. Do metrics agree on the ranking?
    5. Confidence intervals

Outputs are written to results/ : figures (PNG), tables (CSV), and the raw
rankings in TREC format (results/runs/).
"""

import csv
import os
import sys

from analysis import (best_and_worst, bm25_parameter_study, lm_lambda_study,
                      query_features, spearman, spearman_permutation_p, split_by_length,
                      win_tie_loss)
from metrics import NDCG_CUTOFFS, P_CUTOFFS, R_CUTOFFS, evaluate_run, metric_names
from model_suite import MODEL_DESCRIPTIONS, MODEL_ORDER, build_all_runs
from plots import (plot_bm25_heatmaps, plot_lambda_sweep, plot_metrics_vs_k,
                   plot_model_comparison, plot_pr_curves, plot_per_query_scatter,
                   plot_length_effect, plot_query_properties)
from runs import oracle_run, write_trec_qrels, write_trec_run
from significance import bootstrap_ci, holm_correction, paired_permutation_test
from system_builder import build_system

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

K1_VALUES = [0.5, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0]
B_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
PARAM_STUDY_METRICS = ["AP", "nDCG@10", "P@10", "R@100"]

# Models compared in the significance tests: everything except the random
# floor, tested against the project's final model.
REFERENCE_MODEL = "BM25"


def heading(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def shorten(text, width):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def display_name(metric):
    """AP and RR are the per-query values; their means are called MAP and MRR."""
    return {"AP": "MAP", "RR": "MRR"}.get(metric, metric)


def print_table(row_labels, columns, get_value, label_width=18, col_width=9):
    """Prints a table: one row per label, one column per metric name."""
    print(f"{'':<{label_width}}" + "".join(f"{display_name(c):>{col_width}}" for c in columns))
    print("-" * (label_width + col_width * len(columns)))
    for label in row_labels:
        print(f"{label:<{label_width}}" + "".join(f"{get_value(label, c):>{col_width}.4f}" for c in columns))


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
def section_setup(documents, queries, qrels, index):
    heading("1. EVALUATION SET-UP")
    print(f"Collection        : CISI, {len(documents)} documents, {index.stats()['vocabulary_size']} index terms")
    print(f"Queries           : {len(queries)} in total, {len(qrels)} with relevance judgments (evaluated)")
    print(f"Judgments         : {sum(len(v) for v in qrels.values())} relevant (query, doc) pairs, binary relevance")
    print("Unjudged docs     : treated as non-relevant (standard convention)")
    print("\nModels evaluated (all share one preprocessing pipeline and one index):")
    for name in MODEL_ORDER:
        print(f"  {name:<10} {MODEL_DESCRIPTIONS[name]}")
    print(f"  {'Oracle':<10} relevant documents first - NOT a model; shows each metric's ceiling")


def section_headline(results, oracle):
    heading("2. HEADLINE METRICS (mean over 76 queries; MAP = mean of AP, MRR = mean of RR)")
    columns = ["P@10", "R@10", "F1@10", "AP", "RR", "nDCG@10", "RPrec"]
    labels = MODEL_ORDER + ["Oracle (ceiling)"]
    lookup = dict(results, **{"Oracle (ceiling)": oracle})
    print_table(labels, columns, lambda m, c: lookup[m]["mean"][c])


def section_cutoffs(results, oracle):
    heading("3. METRICS AT SEVERAL CUTOFFS")
    labels = MODEL_ORDER + ["Oracle (ceiling)"]
    lookup = dict(results, **{"Oracle (ceiling)": oracle})

    print("\nPrecision - of the top k results, how many are relevant?")
    print_table(labels, [f"P@{k}" for k in P_CUTOFFS], lambda m, c: lookup[m]["mean"][c])

    print("\nRecall - what fraction of all relevant documents is in the top k?")
    print_table(labels, [f"R@{k}" for k in R_CUTOFFS], lambda m, c: lookup[m]["mean"][c])

    print("\nnDCG - rank-aware quality of the top k (1.0 = perfect ordering)")
    print_table(labels, [f"nDCG@{k}" for k in NDCG_CUTOFFS], lambda m, c: lookup[m]["mean"][c])

    print("\nSuccess - fraction of queries with at least one relevant document in the top k")
    print_table(labels, ["Success@1", "Success@5", "Success@10"], lambda m, c: lookup[m]["mean"][c],
                col_width=12)


def section_ceilings(results, oracle):
    heading("4. HOW CLOSE IS EACH MODEL TO THE CEILING?")
    print("Many CISI queries have dozens of relevant documents, so some metrics cannot")
    print("reach 1.0 even for a perfect system. Score as a percentage of the ceiling:\n")
    columns = ["P@10", "R@10", "R@100"]
    print(f"{'':<18}" + "".join(f"{c:>20}" for c in columns))
    print(f"{'Oracle (ceiling)':<18}" + "".join(f"{oracle['mean'][c]:>20.4f}" for c in columns))
    print("-" * 78)
    for name in MODEL_ORDER:
        cells = ""
        for c in columns:
            value = results[name]["mean"][c]
            cells += f"{value:>11.4f} ({100 * value / oracle['mean'][c]:>4.0f}%) "
        print(f"{name:<18}{cells}")


def section_confidence(results):
    heading("5. 95% BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples of the 76 queries)")
    summary = {}
    print(f"{'':<12}{'MAP':>28}{'nDCG@10':>28}{'P@10':>22}")
    print("-" * 90)
    for name in MODEL_ORDER:
        summary[name] = {}
        cells = ""
        for metric, width in [("AP", 28), ("nDCG@10", 28), ("P@10", 22)]:
            values = [row[metric] for row in results[name]["per_query"].values()]
            mean, low, high = bootstrap_ci(values)
            summary[name][metric] = (mean, low, high)
            cells += f"{f'{mean:.3f} [{low:.3f}, {high:.3f}]':>{width}}"
        print(f"{name:<12}{cells}")
    print("\nIntervals that overlap heavily warn that the models may not truly differ.")
    return summary


def section_significance(results):
    heading(f"6. STATISTICAL SIGNIFICANCE - each model vs {REFERENCE_MODEL} (paired permutation test)")
    others = [m for m in MODEL_ORDER if m not in ("Random", REFERENCE_MODEL)]
    print(f"{len(others)} comparisons per metric, so p-values are also Holm-adjusted for multiple testing.")
    print("(Random is excluded: it is a floor, not a competitor.)")

    for metric, label in [("AP", "MAP"), ("nDCG@10", "nDCG@10"), ("P@10", "P@10")]:
        reference_scores = [results[REFERENCE_MODEL]["per_query"][q][metric]
                            for q in sorted(results[REFERENCE_MODEL]["per_query"])]
        diffs, raw_p = [], []
        for name in others:
            scores = [results[name]["per_query"][q][metric]
                      for q in sorted(results[name]["per_query"])]
            diff, p = paired_permutation_test(reference_scores, scores)
            diffs.append(diff)
            raw_p.append(p)
        adjusted = holm_correction(raw_p)

        print(f"\n{label}: {REFERENCE_MODEL} minus other model")
        print(f"{'  other model':<16}{'mean diff':>11}{'raw p':>10}{'Holm p':>10}   verdict")
        for name, diff, p, adj in zip(others, diffs, raw_p, adjusted):
            verdict = "significant" if adj < 0.05 else "not significant"
            print(f"  {name:<14}{diff:>+11.4f}{p:>10.4f}{adj:>10.4f}   {verdict}")


def section_per_query(results, queries, qrels, preprocessor):
    heading("7. PER-QUERY ANALYSIS")
    qids = sorted(results[REFERENCE_MODEL]["per_query"])
    ap = {name: [results[name]["per_query"][q]["AP"] for q in qids] for name in MODEL_ORDER}

    ndcg = {name: [results[name]["per_query"][q]["nDCG@10"] for q in qids] for name in MODEL_ORDER}

    print(f"On how many of the {len(qids)} queries does {REFERENCE_MODEL} score higher than another model?")
    print(f"{'  versus':<16}{'AP wins':>9}{'ties':>6}{'losses':>8}    {'nDCG@10 wins':>14}{'ties':>6}{'losses':>8}")
    for name in MODEL_ORDER:
        if name == REFERENCE_MODEL:
            continue
        a = win_tie_loss(ap[REFERENCE_MODEL], ap[name])
        n = win_tie_loss(ndcg[REFERENCE_MODEL], ndcg[name])
        print(f"  {name:<14}{a[0]:>9}{a[1]:>6}{a[2]:>8}    {n[0]:>14}{n[1]:>6}{n[2]:>8}")

    features = query_features(queries, qrels, preprocessor, qids)
    ref_ap = ap[REFERENCE_MODEL]
    correlations = {
        "n_terms": spearman([features[q]["n_terms"] for q in qids], ref_ap),
        "n_relevant": spearman([features[q]["n_relevant"] for q in qids], ref_ap),
    }
    p_values = {
        key: spearman_permutation_p([features[q][key] for q in qids], ref_ap)
        for key in ("n_terms", "n_relevant")
    }
    print(f"\nWhat makes a query easy or hard for {REFERENCE_MODEL}? (Spearman correlation with AP)")
    print(f"  number of query terms         : rho = {correlations['n_terms']:+.3f}   p = {p_values['n_terms']:.4f}")
    print(f"  number of relevant documents  : rho = {correlations['n_relevant']:+.3f}   p = {p_values['n_relevant']:.4f}")
    print("  (a query with more relevant documents is easier for ANY ranking to score well on,")
    print("   so the second correlation reflects the metric as much as the query)")

    # ---- Does the BM25 vs TF-IDF difference depend on query length? ----
    # NOTE: this question arose AFTER inspecting which queries differ most (the
    # largest TF-IDF wins were all very long queries), so it is a post-hoc
    # finding: its p-values are optimistic and it needs confirming elsewhere.
    n_terms = [features[q]["n_terms"] for q in qids]
    print("\nDoes the BM25-vs-TF-IDF gap depend on query length?   (post-hoc; see note in code)")
    length_stats = {}
    for metric, label in [("AP", "AP "), ("nDCG@10", "nDCG@10")]:
        gap = [results["BM25"]["per_query"][q][metric] - results["TF-IDF"]["per_query"][q][metric] for q in qids]
        rho = spearman(n_terms, gap)
        p = spearman_permutation_p(n_terms, gap)
        length_stats[metric] = (rho, p)
        print(f"  {label:<8} gap vs number of query terms: rho = {rho:+.3f}   p = {p:.4f}")

    # Robustness: is the trend carried by just a handful of extreme queries?
    gap_ap = {q: results["BM25"]["per_query"][q]["AP"] - results["TF-IDF"]["per_query"][q]["AP"] for q in qids}
    extreme = sorted(qids, key=lambda q: abs(gap_ap[q]), reverse=True)[:4]
    kept = [q for q in qids if q not in extreme]
    rho_r = spearman([features[q]["n_terms"] for q in kept], [gap_ap[q] for q in kept])
    p_r = spearman_permutation_p([features[q]["n_terms"] for q in kept], [gap_ap[q] for q in kept])
    print(f"  AP, after dropping the 4 queries with the largest gap {sorted(extreme)}: "
          f"rho = {rho_r:+.3f}   p = {p_r:.4f}")

    short_ids, long_ids, median = split_by_length(features, qids)
    print(f"\n  Median query length = {median} index terms")
    print(f"  {'':<12}{'MAP':>26}{'nDCG@10':>26}")
    print(f"  {'':<12}{'short':>13}{'long':>13}{'short':>13}{'long':>13}")
    for model in ("TF-IDF", "BM25", "LM-JM 0.7"):
        cells = ""
        for metric in ("AP", "nDCG@10"):
            for group in (short_ids, long_ids):
                value = sum(results[model]["per_query"][q][metric] for q in group) / len(group)
                cells += f"{value:>13.4f}"
        print(f"  {model:<12}{cells}")
    print(f"  (short = {len(short_ids)} queries, long = {len(long_ids)} queries)")

    best, worst = best_and_worst(results[REFERENCE_MODEL]["per_query"], "AP", n=3)
    for title, group in [("Best", best), ("Worst", worst)]:
        print(f"\n{title} 3 queries for {REFERENCE_MODEL} (by average precision):")
        for qid, value in group:
            print(f"  query {qid:<4} AP={value:.3f}  relevant={features[qid]['n_relevant']:<4} "
                  f"{shorten(queries[qid], 60)}")
    return features, correlations, (short_ids, long_ids, length_stats)


def section_parameters(index, preprocessor, queries, qrels):
    heading("8. PARAMETER SENSITIVITY - does the best setting depend on the metric?")

    grids = bm25_parameter_study(index, preprocessor, queries, qrels,
                                 K1_VALUES, B_VALUES, PARAM_STUDY_METRICS)
    print("BM25: best (k1, b) found for each metric over the whole grid")
    print(f"{'  metric':<12}{'best k1':>9}{'best b':>9}{'score':>9}{'at default':>13}")
    for metric in PARAM_STUDY_METRICS:
        best = max(grids[metric], key=grids[metric].get)
        print(f"  {display_name(metric):<10}{best[0]:>9}{best[1]:>9}{grids[metric][best]:>9.4f}"
              f"{grids[metric][(1.2, 0.75)]:>13.4f}")

    sweep = lm_lambda_study(index, preprocessor, queries, qrels, LAMBDAS, ["AP", "nDCG@10", "P@10", "R@20"])
    print("\nLanguage model: effect of the Jelinek-Mercer weight lambda")
    print(f"{'  lambda':<10}{'MAP':>9}{'nDCG@10':>10}{'P@10':>9}{'R@20':>9}")
    for lam in LAMBDAS:
        row = sweep[lam]
        print(f"  {lam:<8}{row['AP']:>9.4f}{row['nDCG@10']:>10.4f}{row['P@10']:>9.4f}{row['R@20']:>9.4f}")
    best_lam = max(LAMBDAS, key=lambda lam: sweep[lam]["AP"])
    print(f"\n  lambda with highest MAP in the sweep: {best_lam}")
    return grids, sweep


def section_agreement(results):
    heading("9. DO THE METRICS AGREE ON THE ORDER OF THE MODELS? (best first, Random excluded)")
    contenders = [m for m in MODEL_ORDER if m != "Random"]
    for metric in ["AP", "nDCG@10", "P@10", "R@10", "RR", "RPrec"]:
        order = sorted(contenders, key=lambda m: results[m]["mean"][metric], reverse=True)
        print(f"  {display_name(metric):<8}: " + " > ".join(order))


# ----------------------------------------------------------------------
# File output
# ----------------------------------------------------------------------
def save_tables(results, oracle, features, qrels):
    names = metric_names()

    with open(os.path.join(RESULTS_DIR, "summary_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [display_name(n) for n in names])
        for name in MODEL_ORDER:
            writer.writerow([name] + [f"{results[name]['mean'][n]:.6f}" for n in names])
        writer.writerow(["Oracle (ceiling)"] + [f"{oracle['mean'][n]:.6f}" for n in names])

    with open(os.path.join(RESULTS_DIR, "per_query_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "query_id", "n_relevant", "n_query_terms"] + names)
        for name in MODEL_ORDER:
            for qid in sorted(results[name]["per_query"]):
                row = results[name]["per_query"][qid]
                writer.writerow([name, qid, features[qid]["n_relevant"], features[qid]["n_terms"]]
                                + [f"{row[n]:.6f}" for n in names])


def main():
    # Keep output readable on Windows consoles that use a legacy code page.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    documents, queries, qrels, preprocessor, index = build_system(DATA_DIR)

    # Produce the rankings once, then measure them (see runs.py).
    runs = build_all_runs(documents, queries, qrels, preprocessor, index)
    results = {name: evaluate_run(runs[name], qrels) for name in MODEL_ORDER}
    oracle = evaluate_run(oracle_run(qrels, sorted(qrels), [d.doc_id for d in documents]), qrels)

    # Save the rankings in TREC format so any standard tool can re-evaluate them.
    for name in MODEL_ORDER:
        tag = name.replace(" ", "_")
        write_trec_run(runs[name], os.path.join(RESULTS_DIR, "runs", f"{tag}.run"), tag)
    write_trec_qrels(qrels, os.path.join(RESULTS_DIR, "runs", "cisi.qrels"))

    section_setup(documents, queries, qrels, index)
    section_headline(results, oracle)
    section_cutoffs(results, oracle)
    section_ceilings(results, oracle)
    ci_summary = section_confidence(results)
    section_significance(results)
    features, correlations, length_info = section_per_query(results, queries, qrels, preprocessor)
    grids, sweep = section_parameters(index, preprocessor, queries, qrels)
    section_agreement(results)

    # ---- figures and tables ----
    per_query = {name: results[name]["per_query"] for name in MODEL_ORDER}
    plot_pr_curves(results, os.path.join(RESULTS_DIR, "fig1_pr_curves.png"))
    plot_metrics_vs_k(results, os.path.join(RESULTS_DIR, "fig2_metrics_vs_k.png"))
    plot_model_comparison(ci_summary, os.path.join(RESULTS_DIR, "fig3_model_comparison.png"))
    plot_per_query_scatter(per_query, [("TF-IDF", "BM25"), ("LM-JM 0.7", "BM25")],
                           os.path.join(RESULTS_DIR, "fig4_per_query_scatter.png"))
    qids = sorted(per_query[REFERENCE_MODEL])
    plot_query_properties(features, {q: per_query[REFERENCE_MODEL][q]["AP"] for q in qids},
                          correlations, os.path.join(RESULTS_DIR, "fig5_query_properties.png"),
                          model_name=REFERENCE_MODEL)
    plot_bm25_heatmaps(grids, K1_VALUES, B_VALUES, os.path.join(RESULTS_DIR, "fig6_bm25_heatmaps.png"))
    plot_lambda_sweep(sweep, os.path.join(RESULTS_DIR, "fig7_lambda_sweep.png"))
    short_ids, long_ids, length_stats = length_info
    plot_length_effect(features, per_query, short_ids, long_ids, *length_stats["AP"],
                       os.path.join(RESULTS_DIR, "fig8_query_length_effect.png"))
    save_tables(results, oracle, features, qrels)

    print(f"\nFigures, tables and TREC run files saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
