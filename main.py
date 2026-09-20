"""
main.py

<<<<<<< HEAD
Entry point for the Vector Space Model document similarity assignment.

Loads the 5 .txt documents in documents/, cleans them, builds TF-IDF
vectors, computes the cosine similarity matrix, prints/report-friendly
output to the console, and saves a similarity heatmap image.

Usage:
    python main.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import load_and_clean_documents
from vsm import VectorSpaceModel

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
OUTPUT_HEATMAP = os.path.join(os.path.dirname(__file__), "similarity_heatmap.png")


def print_document_summaries(documents):
    print("=== Documents Loaded ===")
    for i, (name, text) in enumerate(documents):
        word_count = len(text.split())
        preview = " ".join(text.split()[:20])
        print(f"[{i}] {name}  ({word_count} words after cleaning)")
        print(f"    preview: {preview}...")
    print()


def print_top_terms(vsm, documents, n=8):
    print(f"=== Top {n} TF-IDF Terms per Document ===")
    for i, (name, _) in enumerate(documents):
        terms = vsm.top_terms(i, n=n)
        term_str = ", ".join(f"{t}({w})" for t, w in terms)
        print(f"[{i}] {name}: {term_str}")
    print()


def print_similarity_matrix(vsm):
    sim = vsm.similarity_matrix()
    names = [n.replace(".txt", "") for n in vsm.doc_names]

    print("=== Cosine Similarity Matrix ===")
    header = " " * 12 + "".join(f"{n[:10]:>12}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row = "".join(f"{sim[i][j]:>12.3f}" for j in range(len(names)))
        print(f"{row_name[:10]:<12}{row}")
    print()
    return sim


def print_ranked_pairs(vsm):
    print("=== Document Pairs Ranked by Similarity ===")
    for name_a, name_b, score in vsm.most_similar_pairs():
        print(f"  {name_a:<10} <-> {name_b:<10}   cosine similarity = {score}")
    print()


def save_heatmap(sim, doc_names, path):
    labels = [n.replace(".txt", "") for n in doc_names]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate each cell with its similarity score
    for i in range(n):
        for j in range(n):
            value = sim[i][j]
            text_color = "white" if value > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title("Cosine Similarity Between Documents (TF-IDF Vector Space Model)")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved similarity heatmap to: {path}\n")


def main():
    documents = load_and_clean_documents(DOCS_FOLDER)
    if len(documents) < 2:
        raise SystemExit("Need at least 2 documents in the documents/ folder.")

    print_document_summaries(documents)

    vsm = VectorSpaceModel()
    vsm.fit(documents)
    print(f"Vocabulary size (after TF-IDF vectorization): {vsm.vocabulary_size()} terms\n")

    print_top_terms(vsm, documents)

    sim = print_similarity_matrix(vsm)
    print_ranked_pairs(vsm)

    save_heatmap(sim, vsm.doc_names, OUTPUT_HEATMAP)
=======
Entry point for the mini Information Retrieval system.

Usage:
    python main.py

Loads all .txt files in the documents/ folder, builds the dictionary and
inverted index, prints them, and then opens an interactive Boolean query
prompt. Type queries like:

    information AND retrieval
    search OR crawler
    NOT stopword
    dictionary AND NOT postings

Type "exit" to quit.
"""

import os
from inverted_index import InvertedIndex
from boolean_retrieval import BooleanRetrieval

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")


def build_system():
    ii = InvertedIndex(use_stopwords=True)
    ii.add_documents_from_folder(DOCS_FOLDER)
    return ii


def run_demo_queries(br):
    demo_queries = [
        "information AND retrieval",
        "search OR crawler",
        "NOT stopword",
        "dictionary AND NOT postings",
    ]
    print("\n=== Demo Queries ===")
    for q in demo_queries:
        results = br.search(q)
        print(f'\nQuery: "{q}"')
        if results:
            for doc_id, name in results:
                print(f"  -> [{doc_id}] {name}")
        else:
            print("  -> No matching documents.")


def interactive_loop(br):
    print("\n=== Interactive Boolean Search ===")
    print('Type a query (e.g. "search AND engine") or "exit" to quit.')
    while True:
        query = input("\nquery> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        try:
            results = br.search(query)
        except ValueError as e:
            print(f"Error: {e}")
            continue

        if results:
            for doc_id, name in results:
                print(f"  -> [{doc_id}] {name}")
        else:
            print("  -> No matching documents.")


def main():
    ii = build_system()

    print("=== Documents Loaded ===")
    ii.print_documents()

    print("\n=== Dictionary (Vocabulary) ===")
    print(ii.get_dictionary())

    print("\n=== Inverted Index ===")
    ii.print_index()

    br = BooleanRetrieval(ii)
    run_demo_queries(br)
    interactive_loop(br)
>>>>>>> def75831f3ac3e4420b77fd88eba405cf4d23114


if __name__ == "__main__":
    main()
