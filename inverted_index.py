"""
inverted_index.py

Builds a dictionary and an inverted index for a small collection of text
documents. Also handles text preprocessing (lowercasing, punctuation
removal, stop word removal).

Author: <your name here>
"""

import os
import re
import string
from collections import defaultdict

# A small, fixed stop word list is used instead of an external NLP library
# so the assignment has no extra dependencies. Feel free to extend this.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "and", "or", "not", "but",
    "with", "as", "by", "from", "this", "that", "these", "those", "it",
    "its", "into", "such", "so", "than", "then", "which", "who", "whom",
    "their", "they", "them", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "about", "over", "under", "between", "also", "there", "here",
}


class InvertedIndex:
    """
    Holds the document collection, the dictionary (vocabulary) and the
    inverted index (term -> sorted list of document IDs containing it).
    """

    def __init__(self, use_stopwords=True):
        self.use_stopwords = use_stopwords

        # doc_id -> original file name (kept for readable output)
        self.doc_names = {}

        # doc_id -> raw text (kept so results can be displayed/inspected)
        self.doc_text = {}

        # term -> set of doc_ids that contain the term (the inverted index)
        self.index = defaultdict(set)

        self._next_doc_id = 0

    # ------------------------------------------------------------------
    # 1. Adding documents
    # ------------------------------------------------------------------
    def add_document(self, text, name=None):
        """
        Add a single document (a string) to the collection and update the
        inverted index incrementally. Returns the assigned doc_id.
        """
        doc_id = self._next_doc_id
        self._next_doc_id += 1

        self.doc_names[doc_id] = name if name else f"doc{doc_id}"
        self.doc_text[doc_id] = text

        for term in self._tokenize(text):
            self.index[term].add(doc_id)

        return doc_id

    def add_documents_from_folder(self, folder_path):
        """
        Convenience method: reads every .txt file in a folder and adds each
        one as a separate document, sorted by file name for reproducibility.
        """
        file_names = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
        )
        for file_name in file_names:
            full_path = os.path.join(folder_path, file_name)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            self.add_document(text, name=file_name)

    # ------------------------------------------------------------------
    # 2. Preprocessing / tokenization
    # ------------------------------------------------------------------
    def _tokenize(self, text):
        """
        Lowercases the text, strips punctuation, splits on whitespace and
        (optionally) removes stop words. Returns a list of tokens.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = re.findall(r"\b[a-z0-9]+\b", text)

        if self.use_stopwords:
            tokens = [t for t in tokens if t not in STOP_WORDS]

        return tokens

    # ------------------------------------------------------------------
    # 3. Dictionary
    # ------------------------------------------------------------------
    def get_dictionary(self):
        """
        Returns the vocabulary (dictionary) as a sorted list of unique terms.
        """
        return sorted(self.index.keys())

    # ------------------------------------------------------------------
    # 4. Inspection helpers
    # ------------------------------------------------------------------
    def get_postings(self, term):
        """Returns the sorted postings list (doc_ids) for a single term."""
        term = term.lower()
        return sorted(self.index.get(term, set()))

    def all_doc_ids(self):
        return set(self.doc_names.keys())

    def print_index(self):
        """Pretty-prints the full inverted index (term -> postings list)."""
        print(f"{'TERM':<15} POSTINGS (doc_ids)")
        print("-" * 40)
        for term in self.get_dictionary():
            postings = self.get_postings(term)
            print(f"{term:<15} {postings}")

    def print_documents(self):
        for doc_id in sorted(self.doc_names):
            print(f"[{doc_id}] {self.doc_names[doc_id]}: {self.doc_text[doc_id].strip()}")
