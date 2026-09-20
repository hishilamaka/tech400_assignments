<<<<<<< HEAD
# Document Similarity via a Vector Space Model (TF-IDF + Cosine Similarity)

A small system that computes pairwise similarity between a collection of
documents using the classic Vector Space Model: each document is
represented as a TF-IDF weighted vector over the shared vocabulary, and
similarity between any two documents is measured with cosine similarity.

## Document collection

5 publicly available articles/reports on climate change, saved as `.txt`
in `documents/`:

| File | Source / type | Topic |
|---|---|---|
| `doc1.txt` | *Global Change Biology* journal review | Climate variability, agriculture & livelihoods |
| `doc2.txt` | *Annual Review of Sociology* | Climate change and society (sociological perspective) |
| `doc3.txt` | *Allergo Journal International* | Climate change and allergies/asthma |
| `doc4.txt` | Book chapter (John Urry) | Climate change and society (social science framing) |
| `doc5.txt` | US EPA fact sheet | Climate change impacts on Guam |

All five share a broad theme (climate change) but differ significantly in
domain (ecology/agriculture, sociology, medicine, social theory, regional
policy), which makes the similarity scores meaningful to compare and
interpret rather than trivially high or low across the board.
=======
# Mini Information Retrieval System

A small Boolean Information Retrieval (IR) system built for a coursework
assignment. It ingests a collection of 8 text documents, builds a
dictionary and inverted index, and supports Boolean (AND / OR / NOT)
retrieval over the collection.

## Features

- **Document ingestion** — reads all `.txt` files from `documents/`.
- **Preprocessing** — lowercasing, punctuation removal, tokenization,
  and stop word removal.
- **Dictionary** — the sorted set of unique terms across the collection.
- **Inverted index** — maps each term to the sorted list of document IDs
  that contain it.
- **Boolean retrieval** — supports `AND`, `OR`, `NOT`, and combinations
  such as `dictionary AND NOT postings`, evaluated left to right.
- **Interactive CLI** — type your own queries after the demo output.
>>>>>>> def75831f3ac3e4420b77fd88eba405cf4d23114

## Project structure

```
<<<<<<< HEAD
vsm-project/
├── documents/            # the 5 source .txt files
├── preprocessing.py       # text loading + cleaning (PDF/journal noise removal, tokenization)
├── vsm.py                 # TF-IDF vectorization + cosine similarity (the Vector Space Model)
├── main.py                # ties it together: prints results, saves heatmap
├── similarity_heatmap.png # generated similarity matrix visualization
├── requirements.txt
└── README.md
```

## How it works

1. **Loading & cleaning** (`preprocessing.py`) — each `.txt` file is read
   and passed through a cleanup pass that strips common PDF/journal
   extraction artifacts (download banners, running headers, bare page
   numbers, DOI lines, publisher boilerplate) that would otherwise
   distort term frequencies.
2. **Vectorization** (`vsm.py`) — `scikit-learn`'s `TfidfVectorizer`
   converts the cleaned documents into TF-IDF weighted vectors over the
   shared vocabulary (stop words removed, letters-only tokens,
   sublinear term-frequency scaling to reduce the effect of document
   length).
3. **Similarity** (`vsm.py`) — cosine similarity is computed between
   every pair of document vectors, producing a symmetric N×N similarity
   matrix.
4. **Output** (`main.py`) — prints document summaries, each document's
   top TF-IDF terms (as a sanity check that vectors reflect real
   content), the full similarity matrix, a ranked list of document
   pairs by similarity, and saves a heatmap image.

## Requirements

```bash
pip install -r requirements.txt
```
=======
ir-system/
├── documents/          # 8 sample .txt documents
├── inverted_index.py    # document ingestion, preprocessing, dictionary, index
├── boolean_retrieval.py # AND / OR / NOT query evaluation
├── main.py               # CLI entry point (demo + interactive queries)
├── report/
│   └── REPORT.md         # design & implementation report
└── README.md
```

## Requirements

- Python 3.8+ (standard library only — no external dependencies)
>>>>>>> def75831f3ac3e4420b77fd88eba405cf4d23114

## How to run

```bash
python main.py
```

<<<<<<< HEAD
This prints, in order:
- The 5 loaded documents with word counts and previews.
- The shared vocabulary size after vectorization.
- The top 8 TF-IDF terms per document.
- The full cosine similarity matrix.
- All document pairs ranked from most to least similar.

It also saves `similarity_heatmap.png` — a color-coded visualization of
the similarity matrix.

## Report

See [`report/REPORT.md`](report/REPORT.md) for the full write-up:
design decisions, methodology, snapshots of the code and output, and
interpretation of the similarity results.

## License

Submitted for coursework purposes.
=======
This will:
1. Load and preprocess all documents in `documents/`.
2. Print the loaded documents, the dictionary, and the full inverted index.
3. Run four demo Boolean queries.
4. Drop into an interactive prompt where you can type your own queries,
   e.g.:
   ```
   query> information AND retrieval
   query> search OR crawler
   query> NOT stopword
   query> dictionary AND NOT postings
   ```
   Type `exit` to quit.

## Example query syntax

| Query                          | Meaning                                      |
|---------------------------------|-----------------------------------------------|
| `search AND engine`             | Docs containing both "search" and "engine"    |
| `search OR crawler`             | Docs containing either term                    |
| `NOT stopword`                  | Docs that do NOT contain "stopword"           |
| `dictionary AND NOT postings`   | Docs with "dictionary" but without "postings" |

## Report

See [`report/REPORT.md`](report/REPORT.md) for the full write-up of design
decisions, implementation details, and sample output.

## License

This project is submitted for coursework purposes.
>>>>>>> def75831f3ac3e4420b77fd88eba405cf4d23114
