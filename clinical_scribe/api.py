"""FastAPI service exposing the scribe pipeline. NOT FOR CLINICAL USE."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clinical_scribe.notes import NoteGenerationError
from clinical_scribe.pipeline import DISCLAIMER, ScribePipeline, ScribeResult


class NoteRequest(BaseModel):
    """Body of ``POST /v1/notes``."""

    transcript: str = Field(min_length=1, max_length=200_000)


def create_app(pipeline: ScribePipeline | None = None) -> FastAPI:
    """Build the app; pass a pipeline to inject a specific provider (e.g. in tests)."""
    app = FastAPI(title="clinical-scribe", version="0.1.0", description=DISCLAIMER)
    scribe = pipeline or ScribePipeline()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "provider": scribe.provider.name}

    @app.post("/v1/notes", response_model=ScribeResult)
    def create_note(request: NoteRequest) -> ScribeResult:
        try:
            return scribe.run(request.transcript)
        except NoteGenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
