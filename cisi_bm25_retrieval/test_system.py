"""
test_system.py

Automated checks that the system is correct. Run with:

    python -m unittest test_system -v

The most important test compares the optimised inverted-index BM25 against a
deliberately naive "brute force" implementation written from the formula,
sharing no index code. If the two ever disagree, one of them has a bug.
"""

import math
import os
import unittest
from collections import Counter

from bm25 import BM25
from evaluation import (average_precision, interpolated_precision_curve,
                        paired_permutation_test, precision_at_k,
                        recall_at_k, reciprocal_rank)
from experiments import build_system
from inverted_index import InvertedIndex
from preprocessing import Preprocessor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestPreprocessing(unittest.TestCase):
    def test_pipeline(self):
        pre = Preprocessor()
        self.assertEqual(pre.process("The retrieval of Libraries, and RETRIEVING!"),
                         ["retriev", "librari", "retriev"])

    def test_options_can_be_disabled(self):
        pre = Preprocessor(use_stopwords=False, use_stemming=False)
        self.assertEqual(pre.process("The Libraries"), ["the", "libraries"])


class TestInvertedIndex(unittest.TestCase):
    def test_postings_and_statistics(self):
        index = InvertedIndex()
        index.add_document(1, ["a", "b", "a"])
        index.add_document(2, ["b", "c"])
        index.finalize()
        self.assertEqual(index.get_postings("a"), {1: 2})
        self.assertEqual(index.df("b"), 2)
        self.assertEqual(index.n_docs, 2)
        self.assertAlmostEqual(index.avgdl, 2.5)
        self.assertEqual(index.get_postings("zzz"), {})


class TestMetrics(unittest.TestCase):
    # ranking: relevant docs found at ranks 2 and 5, one relevant doc missed
    ranked = [5, 9, 2, 7, 1]
    relevant = {9, 1, 8}

    def test_values(self):
        self.assertAlmostEqual(precision_at_k(self.ranked, self.relevant, 5), 0.4)
        self.assertAlmostEqual(recall_at_k(self.ranked, self.relevant, 5), 2 / 3)
        self.assertAlmostEqual(average_precision(self.ranked, self.relevant), (1 / 2 + 2 / 5) / 3)
        self.assertAlmostEqual(reciprocal_rank(self.ranked, self.relevant), 0.5)

    def test_interpolated_curve(self):
        # recall reaches 1/3 at precision 1/2, and 2/3 at precision 2/5
        curve = interpolated_precision_curve(self.ranked, self.relevant)
        self.assertEqual([round(x, 2) for x in curve],
                         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0])

    def test_permutation_test_detects_a_clear_difference(self):
        a = [0.9, 0.8, 0.85, 0.9, 0.95, 0.8, 0.9, 0.85]
        b = [x - 0.2 for x in a]
        _, p = paired_permutation_test(a, b, n_permutations=5000)
        self.assertLess(p, 0.05)
        _, p_same = paired_permutation_test(a, a, n_permutations=500)
        self.assertGreater(p_same, 0.5)


class TestBM25AgainstBruteForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.queries, cls.qrels, cls.pre, cls.index = build_system(DATA_DIR)

    def brute_force(self, query, k1, b, k3):
        """BM25 computed straight from the formula, scanning every document."""
        doc_terms = {d.doc_id: self.pre.process(d.title + " " + d.text) for d in self.documents}
        N = len(doc_terms)
        avgdl = sum(len(t) for t in doc_terms.values()) / N
        scores = {}
        for term, qtf in Counter(self.pre.process(query)).items():
            n = sum(1 for t in doc_terms.values() if term in t)
            if n == 0:
                continue
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            q_weight = 1.0 if k3 is None else (k3 + 1) * qtf / (k3 + qtf)
            for doc_id, terms in doc_terms.items():
                tf = terms.count(term)
                if tf:
                    norm = 1 - b + b * len(terms) / avgdl
                    scores[doc_id] = scores.get(doc_id, 0.0) + \
                        q_weight * idf * tf * (k1 + 1) / (tf + k1 * norm)
        return scores

    def test_matches_brute_force(self):
        for k1, b, k3 in [(1.2, 0.75, 8.0), (2.0, 0.3, None)]:
            bm25 = BM25(self.index, k1=k1, b=b, k3=k3)
            for qid in (1, 3, 6):
                mine = bm25.score_query(self.pre.process(self.queries[qid]))
                reference = self.brute_force(self.queries[qid], k1, b, k3)
                self.assertEqual(set(mine), set(reference))
                for doc_id in reference:
                    self.assertAlmostEqual(mine[doc_id], reference[doc_id], places=9)

    def test_results_are_sorted_best_first(self):
        bm25 = BM25(self.index)
        scores = [s for _, s in bm25.search(self.pre.process(self.queries[3]))]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestDataset(unittest.TestCase):
    def test_collection_size_and_alignment(self):
        documents, queries, qrels, _, _ = build_system(DATA_DIR)
        self.assertEqual(len(documents), 1460)
        self.assertEqual(len(queries), 112)
        self.assertEqual(len(qrels), 76)
        doc_ids = {d.doc_id for d in documents}
        for relevant in qrels.values():
            self.assertTrue(relevant <= doc_ids)


if __name__ == "__main__":
    unittest.main()
