"""Loader for the synthetic, gold-annotated visit transcripts in ``eval/data``.

File format (one or more cases per ``*.txt`` file; ``#`` lines are comments)::

    === id: c01 | specialty: family medicine | split: dev | codes: J02.0, J02.9
    H X | Visit date: {{DATE|04/12/2091}}  MRN: {{MRN|ZQ-000417}}
    D S | You mentioned the pain started on {{DATE|Monday the 3rd}}.

Each line is one sentence: ``<speaker> <section> | <text>``. Speakers are ``D`` (doctor),
``N`` (nurse), ``P`` (patient or caregiver) and ``H`` (header/signature, no speaker label).
Sections are ``S``/``O``/``A``/``P`` or ``X`` (small talk, questions, headers).
``{{TYPE|text}}`` marks a gold PHI span. Labelling guide:

* S - symptoms and history as reported (including pertinent negatives and medication history).
* O - vitals, examination findings, test results.
* A - the clinician's impression or diagnosis.
* P - orders, prescriptions, counselling, referrals and follow-up.
* X - greetings, clinician questions, acknowledgements, headers and signatures.

All content is synthetic. Every name, date, phone number and ID is invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from clinical_scribe.notes import Section
from clinical_scribe.text import Role

DATA_DIR = Path(__file__).parent / "data"

_SPEAKERS: dict[str, tuple[str, Role]] = {
    "D": ("Doctor: ", Role.CLINICIAN),
    "N": ("Nurse: ", Role.CLINICIAN),
    "P": ("Patient: ", Role.PATIENT),
    "H": ("", Role.UNKNOWN),
}
_MARKUP = re.compile(r"\{\{([A-Z]+)\|([^}]+)\}\}")
_HEADER = re.compile(r"^===\s*(.+)$")
_LINE = re.compile(r"^([DNPH])\s+([SOAPX])\s*\|\s?(.*)$")


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """A gold PHI span with offsets into :attr:`Case.transcript`."""

    type: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class GoldSentence:
    """One transcript line with its speaker role and gold SOAP section."""

    role: Role
    text: str
    section: Section


@dataclass(slots=True)
class Case:
    """A synthetic visit with gold PHI spans, sentence labels and ICD-10 codes."""

    case_id: str
    specialty: str
    split: str
    codes: list[str]
    transcript: str = ""
    phi: list[GoldSpan] = field(default_factory=list)
    sentences: list[GoldSentence] = field(default_factory=list)


def _add_line(case: Case, speaker: str, section: str, body: str) -> None:
    prefix, role = _SPEAKERS[speaker]
    if case.transcript:
        case.transcript += "\n"
    case.transcript += prefix
    plain = ""
    cursor = 0
    for m in _MARKUP.finditer(body):
        plain += body[cursor : m.start()]
        start = len(case.transcript) + len(plain)
        plain += m.group(2)
        case.phi.append(GoldSpan(m.group(1), start, start + len(m.group(2))))
        cursor = m.end()
    plain += body[cursor:]
    case.transcript += plain
    case.sentences.append(GoldSentence(role, plain, Section(section)))


def parse_cases(text: str) -> list[Case]:
    """Parse every case in one annotated file."""
    cases: list[Case] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        header = _HEADER.match(line)
        if header:
            fields = dict(part.split(":", 1) for part in header.group(1).split("|"))
            meta = {k.strip(): v.strip() for k, v in fields.items()}
            cases.append(
                Case(
                    case_id=meta["id"],
                    specialty=meta["specialty"],
                    split=meta["split"],
                    codes=[c.strip() for c in meta["codes"].split(",")],
                )
            )
            continue
        body = _LINE.match(line)
        if body is None or not cases:
            raise ValueError(f"Malformed annotation line: {line!r}")
        _add_line(cases[-1], body.group(1), body.group(2), body.group(3))
    return cases


def load_cases(data_dir: Path = DATA_DIR) -> list[Case]:
    """Load all cases from ``data_dir``, sorted by id."""
    cases: list[Case] = []
    for path in sorted(data_dir.glob("*.txt")):
        cases.extend(parse_cases(path.read_text(encoding="utf-8")))
    return sorted(cases, key=lambda c: c.case_id)
