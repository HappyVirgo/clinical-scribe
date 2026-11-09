"""Tests for transcript parsing, the rule-based provider and the Claude provider (mocked)."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from clinical_scribe.notes import (
    ClaudeNoteProvider,
    NoteGenerationError,
    RuleBasedNoteProvider,
    Section,
    SOAPNote,
    classify_sentence,
    extract_follow_up,
    extract_medications,
)
from clinical_scribe.text import Role, Sentence, parse_transcript, split_sentences

TRANSCRIPT = """Doctor: Good morning.
Patient: I've had a sore throat for three days. It hurts to swallow.
Nurse: Temperature is 101.3 degrees.
Doctor: The rapid strep test came back positive.
Doctor: This is most likely strep throat.
Doctor: I'll prescribe amoxicillin 500 mg twice a day. Come back in one week if not better.
Patient: Thanks."""


def test_split_sentences_respects_abbreviations() -> None:
    assert split_sentences("Dr. [NAME] is here. She will see you.") == [
        "Dr. [NAME] is here.",
        "She will see you.",
    ]


def test_parse_transcript_roles() -> None:
    sentences = parse_transcript("Doctor: Hi.\nPt: Hello. I hurt.\nVisit date: [DATE:2091]")
    assert [s.role for s in sentences] == [Role.CLINICIAN, Role.PATIENT, Role.PATIENT, Role.UNKNOWN]


@pytest.mark.parametrize(
    ("role", "text", "expected"),
    [
        (Role.PATIENT, "My knee has been hurting for two weeks.", Section.SUBJECTIVE),
        (Role.PATIENT, "Okay, thank you.", Section.OTHER),
        (Role.PATIENT, "Is it serious?", Section.OTHER),
        (Role.CLINICIAN, "Any fever?", Section.OTHER),
        (Role.CLINICIAN, "Blood pressure is 152/96 and heart rate is 74.", Section.OBJECTIVE),
        (Role.CLINICIAN, "On exam the lungs are clear.", Section.OBJECTIVE),
        (Role.CLINICIAN, "This is consistent with generalized anxiety disorder.", Section.ASSESSMENT),
        (Role.CLINICIAN, "This looks like shingles.", Section.ASSESSMENT),
        (Role.CLINICIAN, "I'll start amlodipine 5 mg once a day.", Section.PLAN),
        (Role.CLINICIAN, "Drink plenty of fluids.", Section.PLAN),
        (Role.CLINICIAN, "You mentioned the pain started on Monday.", Section.SUBJECTIVE),
    ],
)
def test_classify_sentence(role: Role, text: str, expected: Section) -> None:
    assert classify_sentence(Sentence(role, text)) is expected


def test_rule_based_note_sections() -> None:
    note = RuleBasedNoteProvider().generate(TRANSCRIPT)
    assert note.subjective == ["I've had a sore throat for three days.", "It hurts to swallow."]
    assert note.objective == ["Temperature is 101.3 degrees.", "The rapid strep test came back positive."]
    assert note.assessment == ["This is most likely strep throat."]
    assert len(note.plan) == 2
    assert note.medications[0].model_dump() == {"name": "amoxicillin", "dose": "500 mg", "frequency": "twice a day"}
    assert note.follow_up == "Come back in one week if not better."


def test_rule_based_empty_transcript() -> None:
    note = RuleBasedNoteProvider().generate("")
    assert note == SOAPNote(subjective=[], objective=[], assessment=[], plan=[], medications=[], follow_up=None)


def test_extract_medications_dedupes_and_handles_missing_dose() -> None:
    meds = extract_medications(["Continue metformin.", "Increase metformin to 1000 mg daily."])
    assert [(m.name, m.dose, m.frequency) for m in meds] == [("metformin", None, None)]


def test_extract_follow_up_none() -> None:
    assert extract_follow_up(["Take ibuprofen as needed."]) is None


class _FakeMessages:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.kwargs: dict[str, Any] = {}

    def parse(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def _provider(messages: _FakeMessages) -> ClaudeNoteProvider:
    return ClaudeNoteProvider(client=SimpleNamespace(messages=messages))  # type: ignore[arg-type]


NOTE = SOAPNote(subjective=["sore throat"], objective=[], assessment=["strep"], plan=[], medications=[], follow_up=None)


def test_claude_provider_request_shape_and_result() -> None:
    fake = _FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=NOTE))
    assert _provider(fake).generate("Patient: [NAME] has a sore throat.") == NOTE
    kw = fake.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["output_format"] is SOAPNote
    assert kw["output_config"] == {"effort": "medium"}
    assert kw["extra_body"] == {"fallbacks": "default"}
    assert "temperature" not in kw
    assert "[NAME]" in kw["messages"][0]["content"]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_claude_provider_bad_stop_reasons(stop_reason: str) -> None:
    fake = _FakeMessages(SimpleNamespace(stop_reason=stop_reason, parsed_output=None))
    with pytest.raises(NoteGenerationError):
        _provider(fake).generate("x")


def test_claude_provider_missing_parsed_output() -> None:
    fake = _FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=None))
    with pytest.raises(NoteGenerationError, match="no structured note"):
        _provider(fake).generate("x")


_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (anthropic.RateLimitError("slow", response=httpx.Response(429, request=_REQUEST), body=None), "rate limit"),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=_REQUEST), body=None), "500"),
        (anthropic.APIConnectionError(request=_REQUEST), "reach"),
    ],
)
def test_claude_provider_sdk_errors(error: Exception, message: str) -> None:
    with pytest.raises(NoteGenerationError, match=message):
        _provider(_FakeMessages(error=error)).generate("x")
