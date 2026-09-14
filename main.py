"""
main.py

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


if __name__ == "__main__":
    main()
