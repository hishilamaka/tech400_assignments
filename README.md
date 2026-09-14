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

## Project structure

```
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

## How to run

```bash
python main.py
```

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
