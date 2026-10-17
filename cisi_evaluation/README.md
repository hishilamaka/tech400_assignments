# Evaluating Retrieval Models on CISI: Precision, Recall, MAP, nDCG and More

This project evaluates information retrieval models on the CISI test collection
(assignment 6's BM25 system, extended). It implements the evaluation metrics
from their definitions, applies them to six rankings, and analyses the results
statistically instead of only comparing averages.

## What is evaluated

All models share one preprocessing pipeline (lowercase, tokenize, 135 stop
words, Porter stemming) and one inverted index, so any score difference comes
from the ranking function alone.

| Model | Description |
|---|---|
| Random | Documents in random order (seeded): the floor of every scale |
| TF-IDF | TF-IDF weights + cosine similarity (vector space model) |
| BM25-set | BM25 (k1=1.2, b=0.75) with the query treated as a set of terms |
| **BM25** | BM25 (k1=1.2, b=0.75, k3=8): the final model of assignment 6 |
| LM-JM 0.1 | Query-likelihood language model, Jelinek-Mercer smoothing, lambda = 0.1 |
| LM-JM 0.7 | Same, lambda = 0.7 |

The two lambda values were fixed before looking at any scores, from the
published guidance of Zhai & Lafferty (2001) that verbose queries (as in CISI)
prefer larger lambda. A full lambda sweep is reported separately.

An **Oracle** run (relevant documents ranked first) is not a model. It shows the
**ceiling** of each metric: because many CISI queries have dozens of relevant
documents, even a perfect system scores only 0.454 on Recall@10.

## Metrics (all in `metrics.py`)

| Metric | Question it answers |
|---|---|
| Precision@k | Of the top k results, how many are relevant? |
| Recall@k | What fraction of all relevant documents is in the top k? |
| F1@k | Harmonic mean of Precision@k and Recall@k |
| Success@k | Does the top k contain at least one relevant document? |
| AP / **MAP** | Precision at each relevant document's rank, averaged (rewards early AND complete retrieval) |
| RR / **MRR** | 1 / rank of the first relevant document |
| R-Precision | Precision at rank R, where R = number of relevant documents |
| **nDCG@k** | Rank-discounted gain, normalised by the ideal ranking (1.0 = perfect) |
| 11-point PR curve | Interpolated precision at recall 0.0, 0.1, ... 1.0 |

CISI relevance is binary, so nDCG here uses gains of 0 and 1; it measures how
early relevant documents appear, not graded relevance.

## Project structure

| File | Role |
|---|---|
| `dataset_loader.py`, `preprocessing.py`, `inverted_index.py` | Data, text pipeline, index (from assignment 6; the index now also stores collection frequencies) |
| `bm25.py`, `tfidf_baseline.py`, `lm_jelinek_mercer.py` | The three ranking models, implemented from scratch |
| `system_builder.py`, `model_suite.py` | Build the shared index; define and run all models |
| `runs.py` | Rankings ("runs"), random and oracle runs, TREC-format output |
| `metrics.py` | Every evaluation metric |
| `significance.py` | Bootstrap confidence intervals, permutation test, Holm correction |
| `analysis.py` | Win/tie/loss, Spearman correlation, parameter studies |
| `plots.py` | Figures |
| `main.py` | Runs the whole evaluation and prints the report-ready tables |
| `test_evaluation.py` | 23 automated tests |
| `verify_with_trec_eval.py` | Optional cross-check against `trec_eval` |

## How to run (VS Code, Windows PowerShell)

```powershell
cd path\to\cisi-evaluation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py                         # full evaluation; saves figures/tables to results/
python -m unittest test_evaluation -v  # run the tests
```

Optional independent check of every metric against the reference tool:

```powershell
pip install pytrec_eval-terrier        # needs the Microsoft C++ Build Tools on Windows
python verify_with_trec_eval.py
```

## Results (76 judged queries)

| Model | P@10 | R@10 | F1@10 | MAP | MRR | nDCG@10 | R-Prec |
|---|---|---|---|---|---|---|---|
| Random | 0.0276 | 0.0059 | 0.0094 | 0.0317 | 0.0731 | 0.0239 | 0.0246 |
| TF-IDF | 0.3355 | 0.1403 | 0.1637 | **0.2244** | 0.6319 | 0.3791 | **0.2428** |
| BM25-set | 0.3039 | 0.1115 | 0.1337 | 0.1768 | 0.5914 | 0.3341 | 0.2097 |
| **BM25** | **0.3579** | 0.1393 | 0.1668 | 0.2194 | **0.6532** | **0.3984** | 0.2388 |
| LM-JM 0.1 | 0.3276 | 0.1235 | 0.1483 | 0.1914 | 0.5842 | 0.3601 | 0.2127 |
| LM-JM 0.7 | 0.3539 | **0.1472** | **0.1727** | 0.2130 | 0.6187 | 0.3884 | 0.2381 |
| *Oracle (ceiling)* | 0.9500 | 0.4538 | 0.5208 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Findings

1. **The models are far above chance but far from the ceiling.** Random P@10
   is 0.028, matching the analytic expectation (3114 / (76 x 1460) = 0.028). BM25
   reaches 38% of the P@10 ceiling, 31% of the R@10 ceiling and 46% of R@100.
2. **TF-IDF, BM25 and LM-JM 0.7 are statistically indistinguishable.** Against
   BM25, none of them differs significantly on MAP, nDCG@10 or P@10 (Holm-adjusted
   p between 0.27 and 0.81), and their 95% bootstrap intervals overlap almost
   completely (MAP intervals are roughly +/-0.03 to +/-0.04 wide). BM25-set and LM-JM 0.1 are
   significantly worse (Holm p <= 0.015 on all three metrics).
3. **The choice of metric changes who "wins".** TF-IDF is first on MAP and
   R-Precision, BM25 on P@10, nDCG@10 and MRR, LM-JM 0.7 on R@10. Because the
   underlying differences are not significant, this is noise reshuffling the
   order, not a real difference between the models. It does show that reporting a
   single metric can manufacture a winner.
4. **The BM25-vs-TF-IDF gap depends on query length** (post-hoc finding): BM25 is
   better on short queries (MAP 0.203 vs 0.184, nDCG@10 0.381 vs 0.317) and
   TF-IDF is better on long ones (MAP 0.267 vs 0.236, nDCG@10 0.445 vs 0.417).
   The gap and query length are correlated (Spearman rho = -0.31, p = 0.007 for AP;
   -0.25, p = 0.033 for nDCG@10). The two opposite effects roughly cancel in the
   overall average. The trend weakens to rho = -0.24, p = 0.045 after dropping the
   four queries with the largest gaps, so it is modest and partly driven by a few
   extreme queries. The cause was not investigated.
5. **The best parameters depend on the metric.** The best BM25 (k1, b) is
   (5.0, 0.75) for MAP, (3.0, 0.5) for nDCG@10 and (8.0, 0.75) for P@10, but all
   are broad plateaus and the gain over the defaults is only about 0.01
   (0.008 to 0.012). These are
   in-sample optima, so they are optimistic (assignment 6 showed tuning did not
   generalize to unseen queries).
6. **Lambda matters, and metrics disagree on it too.** MAP rises with lambda to
   a peak at 0.9 (0.2204) then falls; nDCG@10 and P@10 peak at moderate lambda
   (about 0.4 to 0.6). Heavy smoothing helps deep ranking but not the top of the
   list.
7. **Query properties.** Number of query terms is unrelated to AP (rho = +0.09,
   p = 0.43). Number of relevant documents correlates with AP (rho = +0.34,
   p = 0.003), but that partly reflects the metric: any ranking scores higher
   precision on a query with many relevant documents.

## How the evaluation was verified

* 23 automated tests: metrics against hand-computed values; BM25 and the
  language model against brute-force implementations that recompute the formulas
  for every document (the language-model scorer equals the full query likelihood
  up to a per-query constant).
* Every metric (21 per query, 6 models, 76 queries, 9,576 values) agrees with
  `trec_eval` to within 1e-16 (`verify_with_trec_eval.py`). To make the two tools
  rank identically, each document is given a unique score encoding its rank.
* Rankings are saved in TREC format in `results/runs/` (about 17 MB, so it is
  git-ignored and regenerated by `python main.py`) so any standard tool can
  re-evaluate them.

## Limitations

* 76 queries give wide confidence intervals; small differences cannot be
  resolved.
* The same queries were used to make design choices (query-term frequency in
  assignment 6, the parameter grids here), so reported scores are somewhat
  optimistic. CISI has no separate test set.
* The query-length finding came from inspecting results, so it is post-hoc and its
  p-values are optimistic; it needs confirming on another collection.
* Unjudged documents count as non-relevant, so a relevant-but-unjudged document
  would count against a model.
* Binary relevance means nDCG cannot reward finding highly relevant documents
  before marginally relevant ones.
* CISI is small and old; conclusions may not transfer to other collections.

## Data

The CISI collection was created at the University of Glasgow
(<https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/>): 1,460 documents,
112 queries, 3,114 relevance judgments (76 queries have judgments). It is also
distributed on Kaggle.
