"""
boolean_retrieval.py

Implements Boolean query processing (AND, OR, NOT) on top of an
InvertedIndex object. Supports simple two-term queries as well as
arbitrary queries built as a list of tokens, e.g.:

    "information AND retrieval"
    "search OR crawler"
    "NOT stopword"
    "information AND retrieval AND NOT boolean"

The parser below is intentionally simple (left-to-right evaluation,
no operator precedence/parentheses) which is standard for an
introductory IR assignment.
"""


class BooleanRetrieval:
    def __init__(self, inverted_index):
        self.ii = inverted_index

    # ------------------------------------------------------------------
    # Set operations on postings lists
    # ------------------------------------------------------------------
    def term_docs(self, term):
        """All doc_ids containing `term` (lowercased for matching)."""
        return set(self.ii.get_postings(term.lower()))

    def op_and(self, set_a, set_b):
        return set_a & set_b

    def op_or(self, set_a, set_b):
        return set_a | set_b

    def op_not(self, set_a):
        return self.ii.all_doc_ids() - set_a

    # ------------------------------------------------------------------
    # Query parsing/evaluation
    # ------------------------------------------------------------------
    def evaluate(self, query):
        """
        Evaluates a Boolean query string, e.g. "search AND engine",
        "crawler OR index", "NOT stopword", "a AND NOT b".

        Returns a sorted list of matching doc_ids.
        """
        tokens = query.strip().split()
        if not tokens:
            return []

        # Handle a leading NOT, e.g. "NOT stopword"
        i = 0
        if tokens[i].upper() == "NOT":
            result = self.op_not(self.term_docs(tokens[i + 1]))
            i += 2
        else:
            result = self.term_docs(tokens[i])
            i += 1

        # Process remaining "OPERATOR [NOT] term" groups left to right
        while i < len(tokens):
            operator = tokens[i].upper()
            i += 1

            negate_next = False
            if i < len(tokens) and tokens[i].upper() == "NOT":
                negate_next = True
                i += 1

            if i >= len(tokens):
                raise ValueError("Malformed query: missing operand after operator")

            operand = self.term_docs(tokens[i])
            if negate_next:
                operand = self.op_not(operand)
            i += 1

            if operator == "AND":
                result = self.op_and(result, operand)
            elif operator == "OR":
                result = self.op_or(result, operand)
            else:
                raise ValueError(f"Unknown operator: {operator}")

        return sorted(result)

    def search(self, query):
        """Evaluates the query and returns (doc_id, doc_name) pairs."""
        doc_ids = self.evaluate(query)
        return [(doc_id, self.ii.doc_names[doc_id]) for doc_id in doc_ids]
