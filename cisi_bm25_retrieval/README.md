# BM25 Information Retrieval System on the CISI Collection

A complete, from-scratch information retrieval system: preprocessing, an
inverted index, Okapi BM25 ranking, query processing, and a quantitative
evaluation against human relevance judgments. Everything except the Porter
stemmer (`nltk`) is implemented in this repository - no search library is used.

## Dataset: CISI

The CISI collection is a classic IR test collection from the University of
Glasgow: abstracts of library and information science papers.

| Item | File | Count |
|---|---|---|
| Documents (title, author, abstract) | `data/CISI.ALL` | 1,460 |
| Natural-language queries | `data/CISI.QRY` | 112 |
| Relevance judgments | `data/CISI.REL` | 3,114 relevant (query, doc) pairs |

Facts about the data that shape the design (found by inspecting the files):

* **Relevance is binary.** The 3rd/4th columns of `CISI.REL` are always `0`,
  so they carry no grade; every listed pair is simply "relevant".
* **Only 76 of the 112 queries have judgments.** Evaluation uses those 76.
* Relevant documents per query range from 1 to 155 (median 32), so recall at
  a fixed cut-off such as R@10 is bounded well below 1 by construction.
* Queries are long natural-language sentences, and **50 of the 76 judged
  queries repeat at least one term** (see "Findings").

## Architecture

```
 CISI.ALL ──► dataset_loader ──► preprocessing ──► inverted_index ──┐
 (docs)                         (lowercase, tokenize,   term -> {doc: tf}│
                                 stop words, Porter)    doc lengths, avgdl│
                                                                       ▼
 CISI.QRY ──► (same preprocessing) ──► query terms ──► BM25 ──► ranked list
 (queries)                                           (bm25.py)      │
                                                                    ▼
 CISI.REL ─────────────────────────────────────────────► evaluation.py
 (judgments)                                      P@10, R@10, MAP, MRR, PR curve
```

| File | Role |
|---|---|
| `dataset_loader.py` | Parses the tagged CISI files |
| `preprocessing.py` | Lowercase, tokenize, stop words, Porter stemming (shared by docs and queries) |
| `inverted_index.py` | `term -> {doc_id: tf}`, document lengths, average length |
| `bm25.py` | Okapi BM25 scoring and ranking (term-at-a-time over the postings) |
| `tfidf_baseline.py` | TF-IDF cosine ranker used as a comparison baseline |
| `evaluation.py` | P@k, R@k, AP/MAP, MRR, 11-point PR curve, permutation test |
| `experiments.py` | Index building, preprocessing ablation, parameter study |
| `plots.py` | Report figures |
| `main.py` | Runs everything and prints the report-ready output |
| `search.py` | Interactive search prompt |
| `test_system.py` | Automated tests |

## How to run (VS Code, Windows PowerShell)

```powershell
cd path\to\cisi-bm25
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py                      # full pipeline + evaluation, saves figures to results/
python search.py                    # interactive search
python -m unittest test_system -v   # run the tests
```

If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.
In `search.py`, type free text, or `#3` to run CISI query 3 with relevant
documents marked `[REL]`.

## Method

**Preprocessing** (identical for documents and queries): lowercase -> split
into alphanumeric tokens -> remove 135 stop words -> Porter stemming. Title
and abstract are indexed; author names are not.

**Index**: for each term, the documents containing it with the term frequency.
Collection statistics (N, document frequency, document lengths, average
length) are stored because BM25 needs them.

**Ranking** - Okapi BM25:

```
score(D,Q) = sum over distinct query terms q of
             (k3+1)*qtf / (k3+qtf)  *  IDF(q) * f(q,D)*(k1+1) / ( f(q,D) + k1*(1 - b + b*|D|/avgdl) )

IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )
```

Defaults: `k1 = 1.2`, `b = 0.75`, `k3 = 8`. The `1 +` inside the logarithm keeps
IDF non-negative; the textbook form goes negative for terms found in more than
half the documents. Only documents that contain at least one query term are
scored.

## Results (76 judged queries, `python main.py`)

| Model | P@10 | R@10 | MAP | MRR |
|---|---|---|---|---|
| TF-IDF cosine (baseline) | 0.3355 | 0.1403 | 0.2244 | 0.6319 |
| BM25, query treated as a set (no query tf) | 0.3039 | 0.1115 | 0.1768 | 0.5914 |
| **BM25 (k1=1.2, b=0.75, k3=8)** | **0.3579** | 0.1393 | 0.2194 | **0.6532** |

Preprocessing ablation (BM25, default parameters):

| Stop words | Stemming | Vocabulary | MAP |
|---|---|---|---|
| kept | off | 10,013 | 0.1781 |
| kept | on | 6,173 | 0.1970 |
| removed | off | 9,886 | 0.1927 |
| removed | on | 6,059 | **0.2194** |

Figures: `results/pr_curve.png`, `results/param_heatmap.png`; per-query
numbers: `results/bm25_per_query.csv`.

## Findings (and one correction made along the way)

1. **Query term frequency matters a lot on CISI.** My first version collapsed
   the query into a set of terms and BM25 scored MAP 0.177 - *significantly
   below* the TF-IDF baseline (p < 0.001). After checking the code against an
   independent brute-force implementation (it was correct), I traced the gap to
   the discarded query-term repetition. Restoring the query-term-frequency
   component of the original Okapi formula (`k3`) raised MAP to 0.219
   (+0.043, p < 0.001).
2. **BM25 and TF-IDF are not significantly different in MAP here.** BM25 has the
   higher P@10 and MRR, TF-IDF the higher MAP, but the per-query AP difference
   is only -0.005 (permutation test p = 0.55). The P@10 and MRR differences were
   not tested separately. The data do **not** show BM25 outperforming the
   baseline on CISI.
3. **Stop word removal and stemming both help**, and their combination is best
   (MAP 0.178 -> 0.219).
4. **Parameter tuning does not generalize.** The best (k1, b) on all 76 queries
   is (5.0, 0.75) with MAP 0.227, but when tuning on half the queries and
   testing on the unseen half, the tuned setting scored 0.208 versus 0.211 for
   the defaults. The surface in `param_heatmap.png` is a broad plateau; the
   defaults are a reasonable choice.
5. **Failure analysis (query 14, the lowest-scoring query).** "What future is
   there for automatic medical diagnosis?" has 3 relevant documents, the first
   at rank 79. Its most distinctive stemmed term, `diagnosi`, occurs in **no
   document** of the collection; one relevant document (doc 890) shares no term
   with the query at all, and the other two match only the generic term
   `automat`. The top results are therefore documents about *automatic indexing*.
   This is vocabulary mismatch: a bag-of-words model cannot connect "diagnosis"
   to the documents' actual wording. Other low scorers are very long queries
   (query 101 has about 40 index terms) where a single relevant document is
   diluted among many generic terms. Even so, 70 of the 76 queries have at
   least one relevant document in the top 10, and none has AP = 0.

## Limitations

* The decision to include query-term frequency was made after observing
  results on the same 76 judged queries (CISI has no separate test set), so the
  reported BM25 figure is slightly optimistic. The effect is large and the
  component comes from the original Okapi formulation, but this is a caveat.
  `k3 = 8` was one value from a small check; linear, logarithmic and Okapi-style
  weightings all gave MAP between 0.21 and 0.23.
* Bag-of-words matching only: no synonyms, phrases, or semantics.
* CISI is small (1,460 documents) and old; conclusions may not transfer to
  larger or web-scale collections.

## Data source

The CISI collection was created at the University of Glasgow
(<https://ir.dcs.gla.ac.uk/resources/test_collections/cisi/>) and is also
distributed on Kaggle. The three data files were fetched from a public GitHub
mirror, and their sizes and counts (1,460 documents, 112 queries) match the
published description of the collection.
