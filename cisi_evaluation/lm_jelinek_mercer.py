"""
lm_jelinek_mercer.py

Query-likelihood language model with Jelinek-Mercer smoothing (Zhai & Lafferty,
2001), implemented on the same inverted index as BM25 and TF-IDF.

Idea: every document D defines a probability distribution over words. Rank
documents by how likely that distribution is to generate the query Q:

    P(Q | D) = product over query terms q of  P(q | D)^qtf(q)

Maximum-likelihood estimates give P(q | D) = 0 for any query word missing from
D, which would wipe out the whole product. Jelinek-Mercer smoothing fixes this
by linearly mixing the document model with the collection model:

    P(q | D) = (1 - lambda) * tf(q, D) / |D|  +  lambda * cf(q) / |C|

    tf(q, D)  occurrences of q in document D        |D|  length of D
    cf(q)     occurrences of q in the whole collection
    |C|       total number of terms in the collection
    lambda    smoothing weight in (0, 1)

Small lambda trusts the document's own counts; large lambda leans on the
collection. Zhai & Lafferty found that long, verbose queries (like CISI's)
prefer a LARGER lambda than short keyword queries.

Efficient scoring
-----------------
Taking logs turns the product into a sum. For a query term absent from D the
term is  log(lambda * cf(q)/|C|)  - the same value for every document - so it
cannot change the ranking. Subtracting that constant from every term leaves a
score that is non-zero ONLY for terms that actually occur in D:

    score(D, Q) = sum over query terms q found in D of
                  qtf(q) * log( 1 + (1 - lambda) * tf(q, D)/|D|
                                    / ( lambda * cf(q)/|C| ) )

This ranks documents in exactly the same order as the full query likelihood
(verified against a brute-force implementation in test_evaluation.py), but
touches only the postings of the query terms - the same speed advantage
BM25 enjoys.

Query terms that never occur in the collection (cf = 0) are skipped, because
they would give every document the same log(0) penalty.
"""

import math
from collections import Counter


class LMJelinekMercer:
    def __init__(self, index, lam=0.5):
        if not 0.0 < lam < 1.0:
            raise ValueError("lambda must be strictly between 0 and 1")
        self.index = index
        self.lam = lam

    def search(self, query_terms, top_k=None):
        """Returns a ranked list of (doc_id, score), best first."""
        index, lam = self.index, self.lam
        scores = {}

        # Counter keeps repeated query terms: qtf acts as an exponent on
        # P(q|D) in the likelihood, i.e. a multiplier after taking logs.
        for term, qtf in Counter(query_terms).items():
            postings = index.get_postings(term)
            if not postings:
                continue  # term not in the collection at all: ignore it

            # P(term | collection): the background probability.
            p_collection = index.collection_freq(term) / index.total_terms

            for doc_id, tf in postings.items():
                # P(term | document): the maximum-likelihood estimate.
                p_document = tf / index.doc_len[doc_id]

                # Smoothed probability divided by its value for a document
                # that lacks the term, on the log scale (see module docstring).
                ratio = ((1.0 - lam) * p_document) / (lam * p_collection)
                scores[doc_id] = scores.get(doc_id, 0.0) + qtf * math.log(1.0 + ratio)

        # Best first; ties broken by doc_id so results are deterministic.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k] if top_k else ranked
