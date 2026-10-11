"""
tfidf_baseline.py

A TF-IDF + cosine similarity ranker (the Vector Space Model from the earlier
assignment), implemented on the SAME inverted index as BM25.

It exists purely as a baseline: comparing BM25 against it on identical data,
preprocessing and queries gives a fair, quantitative justification for the
choice of retrieval model.

    weight(t, d) = (1 + ln tf(t, d)) * ln(N / df(t))      (sublinear TF)
    score(d, q)  = cosine( vector(d), vector(q) )
"""

import math
from collections import Counter


class TfidfCosine:
    def __init__(self, index):
        self.index = index
        self._idf = {
            term: math.log(index.n_docs / len(postings))
            for term, postings in index.postings.items()
        }
        self._doc_norm = self._compute_doc_norms()

    def _compute_doc_norms(self):
        squared = {doc_id: 0.0 for doc_id in self.index.doc_len}
        for term, postings in self.index.postings.items():
            idf = self._idf[term]
            for doc_id, tf in postings.items():
                w = (1.0 + math.log(tf)) * idf
                squared[doc_id] += w * w
        return {doc_id: math.sqrt(v) for doc_id, v in squared.items()}

    def search(self, query_terms, top_k=None):
        q_counts = Counter(t for t in query_terms if t in self._idf)
        if not q_counts:
            return []

        q_weights = {t: (1.0 + math.log(c)) * self._idf[t] for t, c in q_counts.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))

        dots = {}
        for term, qw in q_weights.items():
            idf = self._idf[term]
            for doc_id, tf in self.index.get_postings(term).items():
                dw = (1.0 + math.log(tf)) * idf
                dots[doc_id] = dots.get(doc_id, 0.0) + qw * dw

        scores = {
            doc_id: dot / (self._doc_norm[doc_id] * q_norm)
            for doc_id, dot in dots.items()
            if self._doc_norm[doc_id] > 0
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k] if top_k else ranked
