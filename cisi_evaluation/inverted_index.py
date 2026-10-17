"""
inverted_index.py

An inverted index that stores, for every term, the documents it occurs in
together with the term frequency in each document:

    term -> { doc_id: tf, doc_id: tf, ... }

Unlike the Boolean index from the earlier assignment (which only recorded
presence/absence), BM25 needs three extra statistics, all collected here:

    tf(t, d)   term frequency of term t in document d     (the postings)
    df(t)      number of documents containing term t      (len of postings)
    |d|, avgdl length of each document, and the average   (doc_len, avgdl)
"""

from collections import Counter, defaultdict


class InvertedIndex:
    def __init__(self):
        self.postings = defaultdict(dict)   # term -> {doc_id: tf}
        self.doc_len = {}                   # doc_id -> number of indexed terms
        self.n_docs = 0
        self.avgdl = 0.0
        self.total_terms = 0                # |C|: total term occurrences in the collection
        self._collection_freq = {}          # term -> total occurrences in the collection

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------
    def add_document(self, doc_id, terms):
        """Adds one preprocessed document (a list of terms) to the index."""
        counts = Counter(terms)
        for term, tf in counts.items():
            self.postings[term][doc_id] = tf
        self.doc_len[doc_id] = len(terms)

    def finalize(self):
        """Call once after all documents are added to compute corpus stats."""
        self.n_docs = len(self.doc_len)
        self.total_terms = sum(self.doc_len.values())
        self.avgdl = self.total_terms / self.n_docs if self.n_docs else 0.0

        # Collection frequency cf(t) = sum of tf(t, d) over all documents.
        # Used by the language-model ranker to estimate P(t | collection).
        self._collection_freq = {
            term: sum(postings.values()) for term, postings in self.postings.items()
        }

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def df(self, term):
        """Document frequency of a term (0 if the term is not indexed)."""
        return len(self.postings.get(term, {}))

    def collection_freq(self, term):
        """Total occurrences of a term across the whole collection (0 if unseen)."""
        return self._collection_freq.get(term, 0)

    def get_postings(self, term):
        """Returns {doc_id: tf} for a term (empty dict if not indexed)."""
        return self.postings.get(term, {})

    def vocabulary(self):
        return sorted(self.postings)

    # ------------------------------------------------------------------
    # Statistics (used for reporting)
    # ------------------------------------------------------------------
    def stats(self):
        return {
            "documents": self.n_docs,
            "vocabulary_size": len(self.postings),
            "total_terms": sum(self.doc_len.values()),
            "avg_doc_length": round(self.avgdl, 2),
            "min_doc_length": min(self.doc_len.values()),
            "max_doc_length": max(self.doc_len.values()),
            "total_postings": sum(len(p) for p in self.postings.values()),
        }

    def most_frequent_terms(self, n=10):
        """Terms ranked by document frequency (most widespread first)."""
        ranked = sorted(self.postings.items(), key=lambda kv: len(kv[1]), reverse=True)
        return [(term, len(p)) for term, p in ranked[:n]]
