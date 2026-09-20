"""
preprocessing.py

Text loading and cleaning utilities for the Vector Space Model
similarity assignment.

Handles two layers of cleanup:
1. PDF/journal extraction noise (repeated headers, footers, page
   numbers, DOI lines) that commonly appears when text is copy-pasted
   or extracted from academic PDFs.
2. Standard IR preprocessing (lowercasing, punctuation/number removal,
   tokenization, stop word removal) before vectorization.
"""

import os
import re
import string

# Reused from the earlier Boolean retrieval assignment: a small, fixed
# stop word list so the project has no extra NLP dependencies beyond
# scikit-learn for the vectorizer itself.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "and", "or", "not", "but",
    "with", "as", "by", "from", "this", "that", "these", "those", "it",
    "its", "into", "such", "so", "than", "then", "which", "who", "whom",
    "their", "they", "them", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "about", "over", "under", "between", "also", "there", "here",
    "we", "our", "us", "you", "your", "i", "he", "she", "his", "her",
    "if", "while", "when", "where", "how", "what", "some", "more",
    "most", "other", "one", "two", "each", "all", "any", "no", "only",
}

# Lines matching these patterns are almost always extraction artifacts
# rather than real article content, so they are dropped before
# tokenization. This list was built by inspecting the actual PDF/text
# exports used in this assignment (journal download banners, running
# headers, bare page numbers, DOI lines, etc.).
NOISE_LINE_PATTERNS = [
    r"^\s*downloaded from\b",              # journal download banners
    r"^\s*\d+\s*$",                        # bare page numbers
    r"^\s*page\s+\d+\s*$",
    r"\bdoi\s*:",                          # DOI lines (also mid-line)
    r"^\s*https?://\S+\s*$",               # bare URLs on their own line
    r"^\s*©.*$",                           # copyright lines
    r"^\s*ISSN\b",
    r"^\s*Annual Review of \w+",           # journal running headers
    r"^\s*Vol\.\s*\d+",                    # "Vol. 12, pp. ..." headers
    r"onlinelibrary\.wiley\.com",          # Wiley platform boilerplate
    r"^\s*wiley\b",
    r"^\s*global change biology\b",
    r"terms and conditions.*wiley online library",
]
_NOISE_RE = re.compile("|".join(NOISE_LINE_PATTERNS), re.IGNORECASE)


def load_raw_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def strip_extraction_noise(text):
    """
    Removes common PDF/journal extraction artifacts line by line, then
    collapses excess blank lines and whitespace left behind.
    """
    cleaned_lines = []
    for line in text.splitlines():
        if _NOISE_RE.search(line):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)   # collapse blank-line runs
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)   # collapse repeated spaces
    return cleaned.strip()


def tokenize(text, use_stopwords=True):
    """
    Lowercases, strips punctuation/digits, and tokenizes into words.
    Used mainly for inspection/reporting (e.g. top terms per document);
    the actual TF-IDF vectorization in vsm.py uses scikit-learn's own
    tokenizer for consistency with its IDF calculations.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = re.findall(r"\b[a-z]{2,}\b", text)  # drop numbers and 1-letter tokens
    if use_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def load_and_clean_documents(folder_path):
    """
    Reads every .txt file in folder_path, strips extraction noise, and
    returns an ordered list of (filename, cleaned_text) tuples, sorted
    by filename for reproducibility.
    """
    file_names = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
    )
    documents = []
    for name in file_names:
        raw = load_raw_text(os.path.join(folder_path, name))
        cleaned = strip_extraction_noise(raw)
        documents.append((name, cleaned))
    return documents
