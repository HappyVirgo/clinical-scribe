"""End-to-end pipeline: redact PHI, draft the SOAP note, suggest ICD-10 codes."""

from __future__ import annotations

import os
import time
from functools import lru_cache

from pydantic import BaseModel

from clinical_scribe.coding import Bm25CodeSuggester, CodeSuggestion, load_subset
from clinical_scribe.notes import ClaudeNoteProvider, NoteProvider, RuleBasedNoteProvider, SOAPNote
from clinical_scribe.phi import PhiSpan, redact

DISCLAIMER = "Research prototype. NOT FOR CLINICAL USE. Output must be reviewed by a clinician."


class ScribeResult(BaseModel):
    """Everything the pipeline produces for one transcript."""

    redacted_transcript: str
    phi_spans: list[PhiSpan]
    note: SOAPNote
    suggested_codes: list[CodeSuggestion]
    provider: str
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


def default_provider() -> NoteProvider:
    """Use Claude when ``ANTHROPIC_API_KEY`` is set, otherwise the offline rule-based provider.

    ``CLINICAL_SCRIBE_PROVIDER=rules`` forces the offline provider.
    """
    forced = os.environ.get("CLINICAL_SCRIBE_PROVIDER", "").strip().lower()
    if forced != "rules" and os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeNoteProvider()
    return RuleBasedNoteProvider()


@lru_cache(maxsize=1)
def default_suggester() -> Bm25CodeSuggester:
    """Shared BM25 index over the bundled ICD-10-CM subset."""
    return Bm25CodeSuggester(load_subset())


class ScribePipeline:
    """Redact -> note -> codes. The provider only ever sees redacted text."""

    def __init__(
        self,
        provider: NoteProvider | None = None,
        suggester: Bm25CodeSuggester | None = None,
        top_k: int = 3,
    ) -> None:
        self.provider = provider or default_provider()
        self.suggester = suggester or default_suggester()
        self.top_k = top_k

    def run(self, transcript: str) -> ScribeResult:
        """Process one visit transcript and time each stage in milliseconds."""
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        redaction = redact(transcript)
        t1 = time.perf_counter()
        note = self.provider.generate(redaction.text)
        t2 = time.perf_counter()
        codes = self.suggester.suggest(" ".join(note.assessment), k=self.top_k)
        t3 = time.perf_counter()
        timings["phi"] = round((t1 - t0) * 1000, 3)
        timings["note"] = round((t2 - t1) * 1000, 3)
        timings["coding"] = round((t3 - t2) * 1000, 3)
        timings["total"] = round((t3 - t0) * 1000, 3)
        return ScribeResult(
            redacted_transcript=redaction.text,
            phi_spans=redaction.spans,
            note=note,
            suggested_codes=codes,
            provider=self.provider.name,
            timings_ms=timings,
        )
