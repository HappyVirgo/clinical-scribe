"""Tests for the BM25 ICD-10 suggester and its bundled code subset."""

import pytest

from clinical_scribe.coding import Bm25CodeSuggester, IcdCode, load_subset, strip_negated, tokenize


@pytest.fixture(scope="module")
def suggester() -> Bm25CodeSuggester:
    return Bm25CodeSuggester(load_subset())


def test_subset_size_and_uniqueness() -> None:
    codes = load_subset()
    assert 80 <= len(codes) <= 120
    assert len({c.code for c in codes}) == len(codes)
    assert all(c.synonyms for c in codes)


def test_tokenize_stems_and_adds_bigrams() -> None:
    assert tokenize("Kidney stones") == ["kidney", "stone", "kidney stone"]


def test_strip_negated_drops_negated_clause_only() -> None:
    assert "pneumonia" not in strip_negated("Acute bronchitis, no evidence of pneumonia.")
    assert "bronchitis" in strip_negated("Acute bronchitis, no evidence of pneumonia.")


@pytest.mark.parametrize(
    ("assessment", "code"),
    [
        ("This is most likely strep throat.", "J02.0"),
        ("Type 2 diabetes, poorly controlled with hyperglycemia.", "E11.65"),
        ("Type 2 diabetes, well controlled.", "E11.9"),
        ("Acute bronchitis, no pneumonia.", "J20.9"),
        ("This looks like shingles.", "B02.9"),
    ],
)
def test_top1(suggester: Bm25CodeSuggester, assessment: str, code: str) -> None:
    assert suggester.suggest(assessment)[0].code == code


def test_evidence_lists_matched_terms(suggester: Bm25CodeSuggester) -> None:
    top = suggester.suggest("strep throat")[0]
    assert "strep throat" in top.evidence
    assert top.score > 0


def test_negated_only_query_returns_nothing(suggester: Bm25CodeSuggester) -> None:
    assert suggester.suggest("No pneumonia.") == []


def test_empty_and_unknown_queries(suggester: Bm25CodeSuggester) -> None:
    assert suggester.suggest("") == []
    assert suggester.suggest("zzzz qqqq") == []


def test_k_limits_results(suggester: Bm25CodeSuggester) -> None:
    assert len(suggester.suggest("pain", k=2)) == 2


def test_custom_codes() -> None:
    s = Bm25CodeSuggester([IcdCode(code="X1", description="Alpha thing", synonyms=["beta"])])
    assert [c.code for c in s.suggest("beta")] == ["X1"]
