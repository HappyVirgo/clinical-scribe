"""Tests for the FastAPI service, pipeline wiring, CLI and the eval loader (all offline)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clinical_scribe.__main__ import main
from clinical_scribe.api import create_app
from clinical_scribe.notes import NoteGenerationError, RuleBasedNoteProvider, SOAPNote
from clinical_scribe.pipeline import ScribePipeline, default_provider
from eval.dataset import load_cases, parse_cases

EXAMPLE = Path(__file__).parents[1] / "examples" / "visit.txt"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(ScribePipeline(provider=RuleBasedNoteProvider())))


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "provider": "rules"}


def test_post_notes(client: TestClient) -> None:
    response = client.post("/v1/notes", json={"transcript": EXAMPLE.read_text(encoding="utf-8")})
    assert response.status_code == 200
    body = response.json()
    assert "Quibbleworth" not in body["redacted_transcript"]
    assert {s["type"] for s in body["phi_spans"]} >= {"NAME", "DATE", "MRN"}
    assert body["note"]["assessment"] == ["This is most likely strep throat."]
    assert body["suggested_codes"][0]["code"] == "J02.0"
    assert body["provider"] == "rules"
    assert set(body["timings_ms"]) == {"phi", "note", "coding", "total"}
    assert "NOT FOR CLINICAL USE" in body["disclaimer"]


def test_post_notes_validation(client: TestClient) -> None:
    assert client.post("/v1/notes", json={"transcript": ""}).status_code == 422


class _FailingProvider:
    name = "failing"

    def generate(self, redacted_transcript: str) -> SOAPNote:
        raise NoteGenerationError("upstream down")


def test_provider_failure_maps_to_502() -> None:
    app = create_app(ScribePipeline(provider=_FailingProvider()))
    response = TestClient(app).post("/v1/notes", json={"transcript": "Doctor: hi."})
    assert response.status_code == 502
    assert response.json()["detail"] == "upstream down"


class _RecordingProvider(RuleBasedNoteProvider):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, redacted_transcript: str) -> SOAPNote:
        self.seen.append(redacted_transcript)
        return super().generate(redacted_transcript)


def test_provider_only_sees_redacted_text() -> None:
    provider = _RecordingProvider()
    ScribePipeline(provider=provider).run("Doctor: Hello Mr. Quibbleworth, call 555-010-1234.")
    assert provider.seen == ["Doctor: Hello Mr. [NAME], call [PHONE]."]


def test_default_provider_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert default_provider().name == "rules"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    assert default_provider().name == "claude"
    monkeypatch.setenv("CLINICAL_SCRIBE_PROVIDER", "rules")
    assert default_provider().name == "rules"


def test_cli_offline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["note", str(EXAMPLE), "--offline"]) == 0
    out = capsys.readouterr().out
    assert "## Assessment" in out and "J02.0" in out and "Quibbleworth" not in out


def test_eval_loader_offsets_and_splits() -> None:
    cases = load_cases()
    assert 20 <= len(cases) <= 30
    assert {c.split for c in cases} == {"dev", "test"}
    for case in cases:
        assert case.codes and case.sentences
        for span in case.phi:
            assert case.transcript[span.start : span.end].strip()


def test_eval_parser_markup() -> None:
    (case,) = parse_cases("=== id: t | specialty: x | split: dev | codes: A00\nD X | Hi {{NAME|Zed}}.\n")
    assert case.transcript == "Doctor: Hi Zed."
    assert case.transcript[case.phi[0].start : case.phi[0].end] == "Zed"


def test_eval_parser_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_cases("=== id: t | specialty: x | split: dev | codes: A00\nnot a line\n")
