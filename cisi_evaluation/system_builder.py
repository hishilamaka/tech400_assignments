"""
system_builder.py

Loads CISI and builds the preprocessed inverted index that every retrieval
model in this project shares. Using one index for all models guarantees that
any difference in the evaluation scores comes from the ranking function alone,
not from different preprocessing or different data.
"""

from dataset_loader import load_cisi
from inverted_index import InvertedIndex
from preprocessing import Preprocessor


def build_index(documents, preprocessor):
    """
    Indexes title + abstract of every document. Authors are excluded because
    author names rarely occur in the queries and only add noise terms.
    """
    index = InvertedIndex()
    for doc in documents:
        terms = preprocessor.process(doc.title + " " + doc.text)
        index.add_document(doc.doc_id, terms)
    index.finalize()
    return index


def build_system(data_folder):
    """Returns (documents, queries, qrels, preprocessor, index)."""
    documents, queries, qrels = load_cisi(data_folder)
    preprocessor = Preprocessor(use_stopwords=True, use_stemming=True)
    index = build_index(documents, preprocessor)
    return documents, queries, qrels, preprocessor, index
