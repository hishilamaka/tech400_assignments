"""
preprocessing.py

Text preprocessing pipeline shared by documents AND queries. Using exactly
the same pipeline on both sides is essential: a query term can only match an
index term if both were normalised in the same way.

Pipeline:
    1. Lowercasing
    2. Tokenization      - runs of letters/digits (punctuation is dropped)
    3. Stop word removal - common function words carry little meaning
    4. Stemming          - Porter stemmer, so "retrieval", "retrieving" and
                           "retrieved" all map to the same index term
"""

import re
from functools import lru_cache

from nltk.stem import PorterStemmer

# A fixed stop word list is bundled here (rather than using nltk.corpus) so
# the project needs no extra downloads.
STOP_WORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours
yourself yourselves may might must shall via etc ie eg
""".split())

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Preprocessor:
    def __init__(self, use_stopwords=True, use_stemming=True):
        self.use_stopwords = use_stopwords
        self.use_stemming = use_stemming
        self._stemmer = PorterStemmer()

        # Stemming is by far the slowest step, and vocabularies repeat a lot,
        # so cache the result per word.
        self._stem = lru_cache(maxsize=None)(self._stemmer.stem)

    def tokenize(self, text):
        """Lowercase and split into alphanumeric tokens."""
        return _TOKEN_PATTERN.findall(text.lower())

    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in STOP_WORDS]

    def stem(self, tokens):
        return [self._stem(t) for t in tokens]

    def process(self, text):
        """Full pipeline: raw text -> list of index terms."""
        tokens = self.tokenize(text)
        if self.use_stopwords:
            tokens = self.remove_stopwords(tokens)
        if self.use_stemming:
            tokens = self.stem(tokens)
        return tokens

    def trace(self, text):
        """
        Returns the output of every stage, for demonstration/reporting:
        {"tokens": [...], "no_stopwords": [...], "stemmed": [...]}
        """
        tokens = self.tokenize(text)
        no_stop = self.remove_stopwords(tokens)
        stemmed = self.stem(no_stop)
        return {"tokens": tokens, "no_stopwords": no_stop, "stemmed": stemmed}
