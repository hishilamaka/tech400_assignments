"""
vsm.py

Vector Space Model implementation: builds TF-IDF document vectors and
computes pairwise cosine similarity between documents.

Uses scikit-learn's TfidfVectorizer rather than a hand-rolled TF-IDF
calculation, since it is the standard, well-tested implementation of
the vector space model's weighting scheme (Salton & McGill) and keeps
the assignment focused on applying and interpreting the model rather
than reimplementing well-known linear algebra.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import STOP_WORDS


class VectorSpaceModel:
    def __init__(self, max_features=None):
        # min_df=1: keep every term that appears in at least one document
        #   (fine at this small scale of 5 documents).
        # sublinear_tf=True: uses 1 + log(tf) instead of raw tf, which is
        #   the standard refinement to stop very long documents (like the
        #   academic PDFs in this collection) from dominating purely by
        #   repeating common terms more often.
        self.vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),
            lowercase=True,
            token_pattern=r"\b[a-zA-Z]{2,}\b",  # letters only, length >= 2
            sublinear_tf=True,
            max_features=max_features,
        )
        self.doc_names = []
        self.tfidf_matrix = None

    def fit(self, documents):
        """
        documents: list of (name, text) tuples.
        Builds the TF-IDF matrix (rows = documents, columns = vocabulary).
        """
        self.doc_names = [name for name, _ in documents]
        texts = [text for _, text in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self.tfidf_matrix

    def vocabulary_size(self):
        return len(self.vectorizer.get_feature_names_out())

    def top_terms(self, doc_index, n=10):
        """
        Returns the top-n highest-weighted TF-IDF terms for one document,
        useful for sanity-checking that each document vector reflects its
        actual subject matter.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        row = self.tfidf_matrix[doc_index].toarray().flatten()
        top_indices = row.argsort()[::-1][:n]
        return [(feature_names[i], round(float(row[i]), 4)) for i in top_indices if row[i] > 0]

    def similarity_matrix(self):
        """
        Returns the full N x N cosine similarity matrix between all
        document vectors (1.0 on the diagonal, symmetric off-diagonal).
        """
        return cosine_similarity(self.tfidf_matrix)

    def most_similar_pairs(self, top_n=None):
        """
        Returns all document pairs (i < j) with their cosine similarity
        score, sorted from most to least similar.
        """
        sim = self.similarity_matrix()
        n = len(self.doc_names)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((self.doc_names[i], self.doc_names[j], round(float(sim[i][j]), 4)))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n] if top_n else pairs
