"""
dataset_loader.py

Parsers for the three CISI collection files:

    CISI.ALL  - the 1,460 documents (title, author, abstract)
    CISI.QRY  - the 112 natural-language queries
    CISI.REL  - relevance judgments (query_id, doc_id, ...)

File format (SMART-style tagged text):
    .I <id>   start of a record
    .T        title
    .A        author(s)
    .W        abstract / query text
    .X        cross-references (citation links)  -> ignored
    .B .C .K  misc. metadata                      -> ignored

Notes on CISI.REL (verified by inspecting the file):
  * The 3rd and 4th columns are always "0" / "0.000000", so they carry no
    relevance grade. Every listed (query, doc) pair is simply a relevant
    document, i.e. relevance is binary.
  * Only 76 of the 112 queries have at least one judgment. Evaluation must
    therefore be restricted to those 76 queries.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass

_FIELD_MARKER = re.compile(r"^\.([A-Z])\s*$")
_RECORD_MARKER = re.compile(r"^\.I\s+(\d+)\s*$")


@dataclass
class Document:
    doc_id: int
    title: str
    author: str
    text: str  # the abstract (.W field)


def _parse_tagged_file(path):
    """
    Reads a .I/.T/.A/.W style file and returns
    {record_id: {"T": str, "A": str, "W": str}}.
    Only T, A and W are kept; every other field is discarded.
    """
    records = {}
    current_id = None
    field = None
    buffers = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            record_match = _RECORD_MARKER.match(line)
            if record_match:
                if current_id is not None:
                    records[current_id] = {k: " ".join(v).strip() for k, v in buffers.items()}
                current_id = int(record_match.group(1))
                buffers = {"T": [], "A": [], "W": []}
                field = None
                continue

            field_match = _FIELD_MARKER.match(line)
            if field_match:
                field = field_match.group(1)
                continue

            if current_id is not None and field in buffers:
                buffers[field].append(line.strip())

    if current_id is not None:
        records[current_id] = {k: " ".join(v).strip() for k, v in buffers.items()}

    return records


def load_documents(path):
    """Returns a list of Document objects sorted by doc_id."""
    records = _parse_tagged_file(path)
    return [
        Document(doc_id=i, title=r["T"], author=r["A"], text=r["W"])
        for i, r in sorted(records.items())
    ]


def load_queries(path):
    """Returns {query_id: query_text}."""
    records = _parse_tagged_file(path)
    return {i: r["W"] for i, r in sorted(records.items())}


def load_qrels(path):
    """
    Returns {query_id: set(relevant_doc_ids)}.
    Only queries that have at least one judgment appear in the result.
    """
    qrels = defaultdict(set)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                qrels[int(parts[0])].add(int(parts[1]))
    return dict(qrels)


def load_cisi(data_folder):
    """Convenience loader: returns (documents, queries, qrels)."""
    documents = load_documents(os.path.join(data_folder, "CISI.ALL"))
    queries = load_queries(os.path.join(data_folder, "CISI.QRY"))
    qrels = load_qrels(os.path.join(data_folder, "CISI.REL"))
    return documents, queries, qrels
