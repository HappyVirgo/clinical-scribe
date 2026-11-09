"""ICD-10-CM code suggestion by BM25 over a curated code subset.

The bundled ``data/icd10_subset.json`` is a small, hand-picked SUBSET of common outpatient
ICD-10-CM codes with lay synonyms. Suggestions are ranked evidence for a human coder
(clinical decision support), never final codes. NOT FOR CLINICAL OR BILLING USE.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from importlib import resources

from pydantic import BaseModel

_STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from has have he her his i if in is it its
    me my of on or our she so that the their them they this to was we were will with you
    your unspecified other site specified initial encounter type think likely probably
    looks like sounds seems most mild moderate severe due also which just very
    """.split()
)
_NEGATIONS = re.compile(r"\b(?:no|not|denies|denied|without|negative for|ruled? out|free of|absence of)\b", re.I)
_CLAUSE_BREAK = re.compile(r"[.;:,!?]|\bbut\b|\bhowever\b", re.I)
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(token: str) -> str:
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, stem and stopword-filter ``text``; adds adjacent-word bigrams."""
    words = [_stem(t) for t in _TOKEN.findall(text.lower())]
    words = [w for w in words if w not in _STOPWORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def strip_negated(text: str) -> str:
    """Remove negated clauses ("no signs of pneumonia") so they cannot match codes."""
    kept: list[str] = []
    for clause in re.split(f"({_CLAUSE_BREAK.pattern})", text, flags=re.I):
        match = _NEGATIONS.search(clause)
        kept.append(clause[: match.start()] if match else clause)
    return " ".join(kept)


class IcdCode(BaseModel):
    """One entry of the curated code subset."""

    code: str
    description: str
    synonyms: list[str]


class CodeSuggestion(BaseModel):
    """A ranked ICD-10 candidate with the query terms that matched it."""

    code: str
    description: str
    score: float
    evidence: list[str]


def load_subset() -> list[IcdCode]:
    """Load the ICD-10-CM subset bundled with the package."""
    raw = resources.files("clinical_scribe").joinpath("data/icd10_subset.json").read_text("utf-8")
    return [IcdCode.model_validate(entry) for entry in json.loads(raw)["codes"]]


class Bm25CodeSuggester:
    """Okapi BM25 over each code's description plus synonyms."""

    def __init__(self, codes: Sequence[IcdCode], k1: float = 1.2, b: float = 0.5) -> None:
        self.codes = list(codes)
        self.k1 = k1
        self.b = b
        self._docs = [self._doc_terms(c) for c in self.codes]
        self._avg_len = sum(sum(d.values()) for d in self._docs) / max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @staticmethod
    def _doc_terms(code: IcdCode) -> Counter[str]:
        # Binary term frequency: a term repeated across many synonyms (e.g. "asthma" in
        # every asthma-exacerbation synonym) should not outweigh a shorter, exact entry.
        terms: set[str] = set()
        for phrase in [code.description, *code.synonyms]:
            terms.update(tokenize(phrase))
        return Counter(terms)

    def _score(self, query: Iterable[str], doc: Counter[str]) -> tuple[float, list[str]]:
        length = sum(doc.values())
        score = 0.0
        matched: list[str] = []
        for term in query:
            tf = doc.get(term, 0)
            if not tf:
                continue
            norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self._avg_len))
            score += self._idf[term] * norm
            matched.append(term)
        return score, matched

    def suggest(self, text: str, k: int = 3) -> list[CodeSuggestion]:
        """Rank codes against free text (normally the note's Assessment), best first."""
        query = set(tokenize(strip_negated(text)))
        if not query:
            return []
        ranked: list[CodeSuggestion] = []
        for code, doc in zip(self.codes, self._docs, strict=True):
            score, matched = self._score(sorted(query), doc)
            if score > 0:
                ranked.append(
                    CodeSuggestion(
                        code=code.code,
                        description=code.description,
                        score=round(score, 3),
                        evidence=sorted(matched, key=lambda t: (-len(t), t)),
                    )
                )
        ranked.sort(key=lambda s: (-s.score, s.code))
        return ranked[:k]
