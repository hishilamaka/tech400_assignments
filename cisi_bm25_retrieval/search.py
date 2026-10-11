"""
search.py

Interactive search over the CISI collection using BM25.

Usage:
    python search.py

At the prompt you can:
    * type any free-text query            e.g.  automatic indexing of documents
    * type  #<n>  to run CISI query n     e.g.  #3
      (relevant documents are then marked [REL], using the CISI judgments)
    * type  quit  to leave
"""

import os
import sys

from bm25 import BM25
from experiments import build_system

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOP_K = 10


def shorten(text, width):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def show_results(query_text, bm25, preprocessor, documents, relevant=None):
    terms = preprocessor.process(query_text)
    if not terms:
        print("  (the query has no usable terms after preprocessing)")
        return

    print(f"  index terms: {' '.join(terms)}")
    results = bm25.search(terms, top_k=TOP_K)
    if not results:
        print("  No document contains any of these terms.")
        return

    by_id = {d.doc_id: d for d in documents}
    for rank, (doc_id, score) in enumerate(results, start=1):
        mark = ""
        if relevant is not None:
            mark = "[REL] " if doc_id in relevant else "      "
        doc = by_id[doc_id]
        print(f"  {rank:>2}. {mark}doc {doc_id:<5} score {score:6.3f}  {shorten(doc.title, 55)}")
        print(f"      {shorten(doc.text, 90)}")

    if relevant is not None:
        hits = sum(1 for d, _ in results if d in relevant)
        print(f"  -> {hits}/{TOP_K} of the top results are relevant "
              f"({len(relevant)} relevant docs in total)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Loading CISI and building the index ...")
    documents, queries, qrels, preprocessor, index = build_system(DATA_DIR)
    bm25 = BM25(index, k1=1.2, b=0.75, k3=8.0)
    print(f"Ready: {index.n_docs} documents, {len(index.postings)} index terms.")
    print('Type a query, "#<n>" for CISI query n, or "quit".')

    while True:
        try:
            line = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break

        if line.startswith("#") and line[1:].isdigit():
            qid = int(line[1:])
            if qid not in queries:
                print(f"  CISI has no query number {qid} (valid: 1-{max(queries)}).")
                continue
            print(f"  CISI query {qid}: {shorten(queries[qid], 110)}")
            relevant = qrels.get(qid)
            if relevant is None:
                print("  (this query has no relevance judgments, so results are not marked)")
            show_results(queries[qid], bm25, preprocessor, documents, relevant)
        else:
            show_results(line, bm25, preprocessor, documents)


if __name__ == "__main__":
    main()
