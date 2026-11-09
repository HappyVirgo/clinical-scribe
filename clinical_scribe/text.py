"""Transcript parsing helpers: speaker turns and sentence segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """Who is speaking in a transcript line."""

    CLINICIAN = "clinician"
    PATIENT = "patient"
    UNKNOWN = "unknown"


_SPEAKERS: dict[str, Role] = {
    "doctor": Role.CLINICIAN,
    "dr": Role.CLINICIAN,
    "physician": Role.CLINICIAN,
    "provider": Role.CLINICIAN,
    "clinician": Role.CLINICIAN,
    "nurse": Role.CLINICIAN,
    "rn": Role.CLINICIAN,
    "ma": Role.CLINICIAN,
    "np": Role.CLINICIAN,
    "pa": Role.CLINICIAN,
    "patient": Role.PATIENT,
    "pt": Role.PATIENT,
    "caregiver": Role.PATIENT,
    "parent": Role.PATIENT,
    "mother": Role.PATIENT,
    "father": Role.PATIENT,
}

_SPEAKER_RE = re.compile(rf"^\s*({'|'.join(_SPEAKERS)})\.?\s*:\s*(.*)$", re.I)
_ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "e.g.", "i.e.", "vs.", "approx.", "St.")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


@dataclass(frozen=True, slots=True)
class Sentence:
    """A single sentence together with the role of whoever said it."""

    role: Role
    text: str


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, without breaking on common abbreviations."""
    pieces = _SENTENCE_END.split(text.strip())
    sentences: list[str] = []
    for piece in pieces:
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        elif piece:
            sentences.append(piece)
    return sentences


def parse_transcript(transcript: str) -> list[Sentence]:
    """Turn a ``Speaker: text`` transcript into role-tagged sentences.

    Lines without a recognised speaker label (headers, signatures) get ``Role.UNKNOWN``.
    """
    sentences: list[Sentence] = []
    for line in transcript.splitlines():
        if not line.strip():
            continue
        match = _SPEAKER_RE.match(line)
        if match:
            role, body = _SPEAKERS[match.group(1).lower()], match.group(2)
        else:
            role, body = Role.UNKNOWN, line
        sentences.extend(Sentence(role, s) for s in split_sentences(body))
    return sentences
