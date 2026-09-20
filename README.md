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

## Project structure

```
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

## How to run

```bash
python main.py
```

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
