"""
test_evaluation.py

Automated correctness checks. Run with:

    python -m unittest test_evaluation -v

An evaluation is only trustworthy if the metric code is right, so the metrics
are tested against values worked out by hand. The ranking models are tested
against deliberately naive "brute force" versions that recompute the formulas
from scratch for every document and share no index code.
"""

import math
import os
import unittest
from collections import Counter

from analysis import spearman, spearman_permutation_p, split_by_length, win_tie_loss
from bm25 import BM25
from inverted_index import InvertedIndex
from lm_jelinek_mercer import LMJelinekMercer
from metrics import (average_precision, dcg_at_k, evaluate_query, f1_at_k,
                     interpolated_precision_curve, metrics_at_every_k, ndcg_at_k,
                     precision_at_k, r_precision, recall_at_k, reciprocal_rank,
                     success_at_k)
from significance import bootstrap_ci, holm_correction, paired_permutation_test
from system_builder import build_system

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class TestMetricsByHand(unittest.TestCase):
    # Ranking of 5 documents; relevant ones are B, E (found) and F (never retrieved).
    #   rank:  1  2  3  4  5
    #   doc:   A  B  C  D  E
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"B", "E", "F"}

    def test_precision_recall_f1(self):
        self.assertAlmostEqual(precision_at_k(self.ranked, self.relevant, 5), 2 / 5)
        self.assertAlmostEqual(precision_at_k(self.ranked, self.relevant, 2), 1 / 2)
        self.assertAlmostEqual(recall_at_k(self.ranked, self.relevant, 5), 2 / 3)
        p, r = 2 / 5, 2 / 3
        self.assertAlmostEqual(f1_at_k(self.ranked, self.relevant, 5), 2 * p * r / (p + r))

    def test_precision_denominator_is_k_even_for_short_lists(self):
        # Only 2 results returned, 1 relevant: P@10 must be 1/10, not 1/2.
        self.assertAlmostEqual(precision_at_k(["B", "X"], self.relevant, 10), 0.1)

    def test_success_and_reciprocal_rank(self):
        self.assertEqual(success_at_k(self.ranked, self.relevant, 1), 0.0)
        self.assertEqual(success_at_k(self.ranked, self.relevant, 2), 1.0)
        self.assertAlmostEqual(reciprocal_rank(self.ranked, self.relevant), 1 / 2)
        self.assertEqual(reciprocal_rank(["X", "Y"], self.relevant), 0.0)

    def test_average_precision(self):
        # relevant found at ranks 2 and 5 -> precisions 1/2 and 2/5; R = 3
        self.assertAlmostEqual(average_precision(self.ranked, self.relevant), (1 / 2 + 2 / 5) / 3)

    def test_r_precision(self):
        # R = 3, so look at the top 3 (A, B, C): only B is relevant
        self.assertAlmostEqual(r_precision(self.ranked, self.relevant), 1 / 3)

    def test_ndcg(self):
        dcg = 1 / math.log2(3) + 1 / math.log2(6)                  # ranks 2 and 5
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)            # 3 relevant docs, best case
        self.assertAlmostEqual(dcg_at_k(self.ranked, self.relevant, 5), dcg)
        self.assertAlmostEqual(ndcg_at_k(self.ranked, self.relevant, 5), dcg / ideal)

    def test_ndcg_perfect_and_worst(self):
        self.assertAlmostEqual(ndcg_at_k(["B", "E", "F", "A"], self.relevant, 3), 1.0)
        self.assertEqual(ndcg_at_k(["X", "Y", "Z"], self.relevant, 3), 0.0)

    def test_ndcg_ideal_uses_min_of_k_and_relevant_count(self):
        # Only 1 relevant document exists: finding it at rank 1 is perfect even for k=10.
        self.assertAlmostEqual(ndcg_at_k(["B", "X", "Y"], {"B"}, 10), 1.0)

    def test_interpolated_curve(self):
        curve = interpolated_precision_curve(self.ranked, self.relevant)
        # recall 1/3 reached at precision 1/2; recall 2/3 reached at precision 2/5
        self.assertEqual([round(v, 4) for v in curve],
                         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0])
        # An interpolated curve can never increase with recall.
        self.assertEqual(curve, sorted(curve, reverse=True))

    def test_empty_relevant_set_is_safe(self):
        row = evaluate_query(self.ranked, set())
        self.assertTrue(all(v == 0.0 for v in row.values()))

    def test_every_k_curves_match_single_k_functions(self):
        p, r, n = metrics_at_every_k(self.ranked, self.relevant, 8)
        for k in range(1, 9):
            self.assertAlmostEqual(p[k - 1], precision_at_k(self.ranked, self.relevant, k))
            self.assertAlmostEqual(r[k - 1], recall_at_k(self.ranked, self.relevant, k))
            self.assertAlmostEqual(n[k - 1], ndcg_at_k(self.ranked, self.relevant, k))


class TestStatistics(unittest.TestCase):
    def test_holm_correction_known_example(self):
        # sorted p: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> raised to 0.06 (monotone)
        adjusted = holm_correction([0.01, 0.04, 0.03])
        self.assertEqual([round(v, 4) for v in adjusted], [0.03, 0.06, 0.06])

    def test_holm_never_exceeds_one(self):
        self.assertTrue(all(v <= 1.0 for v in holm_correction([0.6, 0.7, 0.9])))

    def test_permutation_test(self):
        a = [0.9, 0.8, 0.85, 0.9, 0.95, 0.8, 0.9, 0.85]
        _, p_clear = paired_permutation_test(a, [x - 0.2 for x in a], n_permutations=4000)
        _, p_same = paired_permutation_test(a, a, n_permutations=500)
        self.assertLess(p_clear, 0.05)
        self.assertGreater(p_same, 0.5)

    def test_bootstrap_interval(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        mean, low, high = bootstrap_ci(values, n_resamples=2000)
        self.assertAlmostEqual(mean, 0.45)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertEqual((mean, low, high), bootstrap_ci(values, n_resamples=2000))  # reproducible

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertEqual(spearman([1, 2, 3], [5, 5, 5]), 0.0)            # constant -> undefined -> 0
        self.assertAlmostEqual(spearman([1, 2, 2, 4], [1, 2, 2, 4]), 1.0)  # ties handled

    def test_spearman_permutation_p(self):
        x = list(range(1, 31))
        self.assertLess(spearman_permutation_p(x, [v * 2 for v in x], n_permutations=2000), 0.01)
        unrelated = [7, 2, 9, 4, 1, 8, 3, 6, 5, 10] * 3
        self.assertGreater(spearman_permutation_p(x, unrelated, n_permutations=2000), 0.05)

    def test_split_by_length(self):
        features = {1: {"n_terms": 3}, 2: {"n_terms": 5}, 3: {"n_terms": 9}, 4: {"n_terms": 12}}
        short, long_, median = split_by_length(features, [1, 2, 3, 4])
        self.assertEqual(median, 9)
        self.assertEqual((short, long_), ([1, 2, 3], [4]))
        self.assertEqual(sorted(short + long_), [1, 2, 3, 4])   # nothing lost or duplicated

    def test_win_tie_loss(self):
        self.assertEqual(win_tie_loss([1, 2, 3, 3], [0, 2, 4, 3]), (1, 2, 1))


class TestIndexStatistics(unittest.TestCase):
    def test_collection_frequency(self):
        index = InvertedIndex()
        index.add_document(1, ["a", "b", "a"])
        index.add_document(2, ["b", "c"])
        index.finalize()
        self.assertEqual(index.collection_freq("a"), 2)
        self.assertEqual(index.collection_freq("b"), 2)
        self.assertEqual(index.collection_freq("zzz"), 0)
        self.assertEqual(index.total_terms, 5)


class TestModelsAgainstBruteForce(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents, cls.queries, cls.qrels, cls.pre, cls.index = build_system(DATA_DIR)
        cls.doc_terms = {d.doc_id: cls.pre.process(d.title + " " + d.text) for d in cls.documents}

    def test_bm25_matches_brute_force(self):
        k1, b, k3 = 1.2, 0.75, 8.0
        docs = self.doc_terms
        N = len(docs)
        avgdl = sum(len(t) for t in docs.values()) / N
        model = BM25(self.index, k1=k1, b=b, k3=k3)

        for qid in (1, 3, 14):
            reference = {}
            for term, qtf in Counter(self.pre.process(self.queries[qid])).items():
                n = sum(1 for t in docs.values() if term in t)
                if n == 0:
                    continue
                idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
                for doc_id, terms in docs.items():
                    tf = terms.count(term)
                    if tf:
                        norm = 1 - b + b * len(terms) / avgdl
                        reference[doc_id] = reference.get(doc_id, 0.0) + \
                            (k3 + 1) * qtf / (k3 + qtf) * idf * tf * (k1 + 1) / (tf + k1 * norm)
            mine = model.score_query(self.pre.process(self.queries[qid]))
            self.assertEqual(set(mine), set(reference))
            for doc_id in reference:
                self.assertAlmostEqual(mine[doc_id], reference[doc_id], places=9)

    def test_language_model_matches_full_query_likelihood(self):
        """
        Compute log P(Q|D) for EVERY document the long way (including the
        smoothed probability for terms a document lacks) and check that the
        efficient scorer equals it up to one query-dependent constant.
        """
        lam = 0.7
        docs = self.doc_terms
        total = sum(len(t) for t in docs.values())
        cf = Counter(term for terms in docs.values() for term in terms)
        model = LMJelinekMercer(self.index, lam=lam)

        for qid in (1, 3, 14):
            q = Counter(t for t in self.pre.process(self.queries[qid]) if cf[t] > 0)

            def full_log_likelihood(terms):
                counts = Counter(terms)
                return sum(
                    qtf * math.log((1 - lam) * counts[t] / len(terms) + lam * cf[t] / total)
                    for t, qtf in q.items()
                )

            # The constant = what a document containing NONE of the terms scores.
            offset = sum(qtf * math.log(lam * cf[t] / total) for t, qtf in q.items())
            mine = dict(model.search(self.pre.process(self.queries[qid])))

            for doc_id, terms in docs.items():
                expected = full_log_likelihood(terms) - offset
                self.assertAlmostEqual(mine.get(doc_id, 0.0), expected, places=9)

    def test_dataset_alignment(self):
        self.assertEqual((len(self.documents), len(self.queries), len(self.qrels)), (1460, 112, 76))
        doc_ids = {d.doc_id for d in self.documents}
        self.assertTrue(all(rel <= doc_ids for rel in self.qrels.values()))


if __name__ == "__main__":
    unittest.main()
