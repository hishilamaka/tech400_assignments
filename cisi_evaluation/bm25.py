"""
bm25.py

Okapi BM25 ranking function, implemented from scratch on top of the
InvertedIndex (no search library is used).

    score(D, Q) = sum over distinct query terms q of:

        (k3 + 1) * qtf(q)        IDF(q) * f(q, D) * (k1 + 1)
        ----------------- *  -------------------------------------------
          k3 + qtf(q)         f(q, D) + k1 * (1 - b + b * |D| / avgdl)

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

where
    f(q, D)  term frequency of q in document D
    |D|      length of D (number of indexed terms)
    avgdl    average document length in the collection
    N        number of documents in the collection
    n(q)     number of documents containing q
    qtf(q)   number of times q occurs in the QUERY
    k1       term-frequency saturation  (typical range 1.2 - 2.0)
    b        length normalisation       (0 = none, 1 = full; typical 0.75)
    k3       query-term-frequency saturation (see below)

Query term frequency (k3): CISI queries are long natural-language sentences
in which repeating a word signals emphasis (e.g. query 1 says "titles"
three times); 50 of the 76 judged queries contain a repeated term. The
original Okapi formulation (Robertson et al., 1996) therefore weights each
query term by its frequency in the query, saturating at large counts. Setting
k3=None disables this and treats the query as a plain SET of terms - kept as
an option so the report can measure how much the component matters.

IDF variant: the "1 +" inside the logarithm is the form used by Lucene. The
textbook form ln((N - n + 0.5)/(n + 0.5)) becomes NEGATIVE for terms that
occur in more than half of all documents, which would wrongly penalise a
document for containing a common query word. Adding 1 keeps IDF >= 0.

Query processing is term-at-a-time: only documents that appear in the
postings of at least one query term are ever scored, which is exactly what
makes an inverted index faster than scanning every document.
"""

import math
from collections import Counter


class BM25:
    def __init__(self, index, k1=1.2, b=0.75, k3=8.0):
        self.index = index
        self.k1 = k1
        self.b = b
        self.k3 = k3  # None -> ignore query term frequency
        self._idf_cache = {}

    def idf(self, term):
        if term not in self._idf_cache:
            n = self.index.df(term)
            N = self.index.n_docs
            self._idf_cache[term] = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        return self._idf_cache[term]

    def _query_weight(self, qtf):
        if self.k3 is None:
            return 1.0
        return (self.k3 + 1.0) * qtf / (self.k3 + qtf)

    def score_query(self, query_terms):
        """
        Scores every document that contains at least one query term.
        Returns {doc_id: bm25_score}.
        """
        scores = {}
        k1, b = self.k1, self.b
        avgdl = self.index.avgdl

        for term, qtf in Counter(query_terms).items():
            postings = self.index.get_postings(term)
            if not postings:
                continue  # term never appears in the collection
            weight = self._query_weight(qtf) * self.idf(term)

            for doc_id, tf in postings.items():
                length_norm = 1.0 - b + b * (self.index.doc_len[doc_id] / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (tf * (k1 + 1.0)) / (tf + k1 * length_norm)

        return scores

    def search(self, query_terms, top_k=None):
        """
        Returns a ranked list of (doc_id, score), best first.
        Ties are broken by doc_id so results are deterministic.
        """
        scores = self.score_query(query_terms)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k] if top_k else ranked
