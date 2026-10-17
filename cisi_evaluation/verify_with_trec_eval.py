"""
verify_with_trec_eval.py

Independent cross-check of every metric in metrics.py against trec_eval, the
reference evaluation tool of the IR community (accessed through its Python
binding, pytrec_eval).

Why: the hand-computed tests in test_evaluation.py prove the metrics match MY
understanding of them. This script proves they match the STANDARD
implementation, query by query, for every model.

This script is OPTIONAL. The rest of the project needs only nltk and
matplotlib. To run it:

    pip install pytrec_eval-terrier
    python verify_with_trec_eval.py

(pytrec_eval compiles a small C extension; on Windows this needs the
"Microsoft C++ Build Tools". If installation fails you can skip this script.)
"""

import os
import sys

from metrics import (F1_CUTOFFS, NDCG_CUTOFFS, P_CUTOFFS, R_CUTOFFS,
                     SUCCESS_CUTOFFS, evaluate_run)
from model_suite import MODEL_ORDER, build_all_runs
from system_builder import build_system

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

try:
    import pytrec_eval
except ImportError:
    print("pytrec_eval is not installed.\n"
          "Install it with:  pip install pytrec_eval-terrier\n"
          "(this check is optional - the rest of the project does not need it)")
    sys.exit(0)


def to_pytrec_run(run):
    """
    Converts a run to pytrec_eval's {query: {doc: score}} format.

    trec_eval ignores our ranks and re-sorts by score, breaking ties by
    document id in a different way than we do. To make both tools rank
    identically, every document gets a UNIQUE score that encodes our exact
    order (best = largest).
    """
    converted = {}
    for qid, ranking in run.items():
        n = len(ranking)
        converted[str(qid)] = {str(doc_id): float(n - i) for i, (doc_id, _) in enumerate(ranking)}
    return converted


def main():
    documents, queries, qrels, preprocessor, index = build_system(DATA_DIR)
    runs = build_all_runs(documents, queries, qrels, preprocessor, index)

    pytrec_qrels = {str(q): {str(d): 1 for d in docs} for q, docs in qrels.items()}
    measures = {
        "map", "recip_rank", "Rprec",
        "P." + ",".join(map(str, P_CUTOFFS)),
        "recall." + ",".join(map(str, R_CUTOFFS)),
        "ndcg_cut." + ",".join(map(str, NDCG_CUTOFFS)),
        "success." + ",".join(map(str, SUCCESS_CUTOFFS)),
    }
    evaluator = pytrec_eval.RelevanceEvaluator(pytrec_qrels, measures)

    # Our metric name -> the trec_eval measure that should equal it.
    mapping = {"AP": "map", "RR": "recip_rank", "RPrec": "Rprec"}
    mapping.update({f"P@{k}": f"P_{k}" for k in P_CUTOFFS})
    mapping.update({f"R@{k}": f"recall_{k}" for k in R_CUTOFFS})
    mapping.update({f"nDCG@{k}": f"ndcg_cut_{k}" for k in NDCG_CUTOFFS})
    mapping.update({f"Success@{k}": f"success_{k}" for k in SUCCESS_CUTOFFS})

    print(f"Comparing {len(mapping)} metrics + F1 for {len(MODEL_ORDER)} models "
          f"on {len(qrels)} queries (per-query values)\n")
    print(f"{'model':<12}{'values compared':>17}{'max abs difference':>22}")
    print("-" * 51)

    overall_worst = 0.0
    for model in MODEL_ORDER:
        ours = evaluate_run(runs[model], qrels)["per_query"]
        theirs = evaluator.evaluate(to_pytrec_run(runs[model]))

        worst, compared = 0.0, 0
        for qid, row in ours.items():
            reference = theirs[str(qid)]
            for our_name, ref_name in mapping.items():
                worst = max(worst, abs(row[our_name] - reference[ref_name]))
                compared += 1
            # trec_eval has no F1; rebuild it from ITS precision and recall.
            for k in F1_CUTOFFS:
                p, r = reference[f"P_{k}"], reference[f"recall_{k}"]
                f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
                worst = max(worst, abs(row[f"F1@{k}"] - f1))
                compared += 1

        overall_worst = max(overall_worst, worst)
        print(f"{model:<12}{compared:>17}{worst:>22.2e}")

    print("-" * 51)
    if overall_worst < 1e-9:
        print("PASS: every metric agrees with trec_eval to within floating-point precision.")
    else:
        print(f"FAIL: largest disagreement {overall_worst:.3e} - investigate before trusting results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
