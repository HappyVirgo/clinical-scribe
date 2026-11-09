"""Deterministic PHI de-identification for free-text visit transcripts.

Covers the HIPAA Safe Harbor identifiers that realistically appear in dictated or
transcribed clinical text: names, dates (year kept), phone/fax numbers, e-mail
addresses, SSNs, medical record / account numbers, street addresses and ZIP codes,
URLs, IP addresses and ages over 89.

The detector is a set of ordered regular expressions plus a few context heuristics
(titles, "patient <Name>", signatures, relatives). It is a research prototype and is
NOT FOR CLINICAL USE: it will miss identifiers that do not follow these patterns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel


class PhiType(StrEnum):
    """Categories of identifiers the redactor emits."""

    NAME = "NAME"
    DATE = "DATE"
    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"
    ACCOUNT = "ACCOUNT"
    ADDRESS = "ADDRESS"
    ZIP = "ZIP"
    URL = "URL"
    IP = "IP"
    AGE = "AGE"


class PhiSpan(BaseModel):
    """One detected identifier, with offsets into the *original* text."""

    type: PhiType
    start: int
    end: int
    text: str
    replacement: str


class RedactionResult(BaseModel):
    """Redacted text plus the spans that were replaced."""

    text: str
    spans: list[PhiSpan]


_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"
_NAME_WORD = r"[A-Z][a-z]+(?:['-][A-Z]?[a-z]+)?"
_FULL_NAME = rf"{_NAME_WORD}(?:\s+[A-Z]\.)?(?:\s+{_NAME_WORD})?"
_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl"
    r"|Terrace|Circle|Parkway|Pkwy|Highway|Hwy)"
)
_STATE = r"[A-Z]{2}"

# Capitalised words that frequently follow our name cues but are not names.
_NOT_NAMES = frozenset(
    """
    The A An And But So If In On At To For Of With From By Is It I We You He She They
    This That These Those My Your His Her Our Their Here There Today Yesterday Tomorrow
    Good Morning Afternoon Evening Hello Hi Thanks Thank Please Yes No Okay Ok Well
    Patient Doctor Nurse Name Date Visit Signed Electronically Room Clinic Hospital Center
    Department Unit Service Office Team Everyone Sir Madam Mom Dad Folks Again Sorry
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday
    January February March April May June July August September October November December
    Dr Mr Mrs Ms Mx Miss Prof MD DO NP PA RN Is Was Has Had Will Can Could Would Should Not
    """.split()
)

# (type, pattern, group-with-the-identifier). Order matters: earlier wins on overlap.
_PATTERNS: list[tuple[PhiType, re.Pattern[str], int]] = [
    (PhiType.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 0),
    (PhiType.URL, re.compile(r"\b(?:https?://|www\.)[^\s,;)]+[^\s,;).]"), 0),
    (PhiType.IP, re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 0),
    (PhiType.SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0),
    (
        PhiType.SSN,
        re.compile(r"\b(?:SSN|social security(?: number)?)\s*(?:is|:|#)?\s*(\d{9})\b", re.I),
        1,
    ),
    (
        PhiType.FAX,
        re.compile(
            r"\bfax(?:\s+[a-z]+){0,3}?\s*(?:is|:|#|at|to)?\s*"
            r"((?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4})",
            re.I,
        ),
        1,
    ),
    (
        PhiType.PHONE,
        re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b"),
        0,
    ),
    (
        PhiType.MRN,
        re.compile(
            r"\b(?:MRN|medical record(?: number)?|record number|chart number)"
            r"\s*(?:is|:|#|no\.?|number)?\s*#?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
            re.I,
        ),
        1,
    ),
    (
        PhiType.ACCOUNT,
        re.compile(
            r"\b(?:account|acct\.?|member id|member number|policy(?: number)?|insurance id"
            r"|plan id|claim(?: number)?)\s*(?:is|:|#|no\.?|number)?\s*#?\s*([A-Z0-9][A-Z0-9-]{3,})\b",
            re.I,
        ),
        1,
    ),
    (
        PhiType.ADDRESS,
        re.compile(
            rf"\b\d{{1,6}}\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET_SUFFIX}\.?"
            rf"(?:,?\s+(?:Apt|Apartment|Unit|Suite|Ste)\.?\s*#?\w+)?"
            rf"(?:,\s*(?:[A-Z][a-z]+\s?){{1,3}}(?=,|\s+{_STATE}\b))?"
        ),
        0,
    ),
    (PhiType.ZIP, re.compile(rf"(?<=\b{_STATE}\s)\d{{5}}(?:-\d{{4}})?\b"), 0),
    (PhiType.ZIP, re.compile(r"\b(?:zip(?: code)?|postal code)\s*(?:is|:)?\s*(\d{5}(?:-\d{4})?)\b", re.I), 1),
    (
        PhiType.ADDRESS,
        re.compile(
            rf"\b(?:lives?|living|resides?|moved|relocated|from)\s+(?:in|to|over in)\s+"
            rf"((?:{_NAME_WORD})(?:\s+{_NAME_WORD})?)"
        ),
        1,
    ),
    (PhiType.DATE, re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"), 0),
    (PhiType.DATE, re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:\d{4}|\d{2})\b"), 0),
    (PhiType.DATE, re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}{_ORD}(?:,?\s+\d{{4}})?\b"), 0),
    (PhiType.DATE, re.compile(rf"\b\d{{1,2}}{_ORD}\s+(?:of\s+)?{_MONTHS}\.?(?:,?\s+\d{{4}})?\b"), 0),
    (PhiType.DATE, re.compile(rf"\b{_MONTHS}\.?\s+(?:of\s+)?\d{{4}}\b"), 0),
    (
        PhiType.AGE,
        re.compile(
            r"\b(?:9\d|1[0-1]\d)(?:\s*|-)(?:years?|yrs?|y/?o)\b(?:[\s-]old)?"
            r"|\b(?:aged?|age of)\s+(?:9\d|1[0-1]\d)\b"
            r"|\b(?:9\d|1[0-1]\d)(?=th birthday)",
            re.I,
        ),
        0,
    ),
]

_NAME_PATTERNS: list[re.Pattern[str]] = [
    # Titles: Mr. / Mrs. / Ms. / Miss / Dr. / Doctor / Prof.
    re.compile(rf"\b(?:Mr|Mrs|Ms|Mx|Miss|Dr|Prof)\.?\s+({_FULL_NAME})"),
    re.compile(rf"\bDoctor\s+({_FULL_NAME})"),
    # "patient Jane Doe", "pt. Jane Doe", "patient name: Jane Doe". A bare "Patient:"
    # is a speaker label, so the colon is only accepted after "name".
    re.compile(rf"\b(?:[Pp]atient|[Pp]t\.?)(?:\s+name\s*:?\s*|\s+)(?:is\s+)?({_FULL_NAME})"),
    re.compile(rf"\b(?:[Nn]ame|NAME)\s*(?:is|:)\s*({_FULL_NAME})"),
    re.compile(rf"\b(?:[Mm]y name is|I'm|I am|[Cc]all me)\s+({_FULL_NAME})"),
    # Signatures and credentials: "Jane Doe, MD", "signed by Jane Doe"
    re.compile(rf"\b({_FULL_NAME}),?\s+(?:MD|M\.D\.|DO|NP|PA-C|PA|RN|APRN|FNP|DPT)\b"),
    re.compile(
        rf"\b(?:[Ss]igned(?: by)?|[Ss]ignature|[Dd]ictated by|[Aa]ttending|[Pp]rovider|[Rr]eferred by)"
        rf"\s*:?\s+(?:Dr\.?\s+)?({_FULL_NAME})"
    ),
    # Greetings and direct address: "Hi Jane,", "Thanks for coming in today, Jane."
    re.compile(rf"\b(?:Hi|Hello|Hey|Welcome(?: back)?)\s*,?\s+({_FULL_NAME})"),
    re.compile(rf",\s+({_NAME_WORD})(?=[.!?]\s*$)", re.M),
    # Relatives and household members: "my wife Jane", "her son Tom", "mom Jane present".
    re.compile(
        rf"\b(?:(?:[Mm]y|[Hh]is|[Hh]er|[Yy]our|[Tt]heir)\s+)?"
        rf"(?:wife|husband|partner|son|daughter|mother|mom|father|dad|sister|brother|"
        rf"grandson|granddaughter|caregiver|friend|neighbor|aunt|uncle)\s*,?\s+({_FULL_NAME})"
    ),
]

# Header/registration lines ("Clinic, Jane Doe, DOB ...") list a full name between
# delimiters with no other cue. Only applied to lines without a "Speaker:" label.
_HEADER_NAME = re.compile(
    rf"(?:^|[,;|—–-]\s*|\bfor\s+)({_NAME_WORD}\s+(?:[A-Z]\.\s+)?{_NAME_WORD})"
    rf"(?=\s*(?:[,;|—–-]|$))"
)
_SPEAKER_LINE = re.compile(r"^\s*[A-Za-z]{1,12}\s*:")

_YEAR = re.compile(r"\b(\d{4})\b")


def _name_spans(text: str) -> Iterable[tuple[PhiType, int, int]]:
    """Find names from context cues, then propagate each name token across the text."""
    tokens: set[str] = set()
    for pattern in _NAME_PATTERNS:
        for m in pattern.finditer(text):
            run: list[re.Match[str]] = []
            for word in re.finditer(_NAME_WORD, m.group(1)):
                if word.group() in _NOT_NAMES:
                    break
                run.append(word)
            if not run:
                continue
            tokens.update(w.group() for w in run)
            yield PhiType.NAME, m.start(1) + run[0].start(), m.start(1) + run[-1].end()
    for line in re.finditer(r"^.*$", text, re.M):
        if _SPEAKER_LINE.match(line.group()):
            continue
        for m in _HEADER_NAME.finditer(line.group()):
            words = m.group(1).split()
            if any(w.rstrip(".") in _NOT_NAMES for w in words):
                continue
            tokens.update(w for w in words if len(w) > 2)
            yield PhiType.NAME, line.start() + m.start(1), line.start() + m.end(1)
    for token in tokens:
        for m in re.finditer(rf"\b{re.escape(token)}\b", text):
            yield PhiType.NAME, m.start(), m.end()


def _resolve_overlaps(cands: list[tuple[int, PhiType, int, int]]) -> list[tuple[PhiType, int, int]]:
    """Keep non-overlapping spans, preferring higher priority, then longer spans."""
    cands.sort(key=lambda c: (c[0], -(c[3] - c[2]), c[2]))
    taken: list[tuple[PhiType, int, int]] = []
    for _, phi_type, start, end in cands:
        if start >= end:
            continue
        if any(start < t_end and t_start < end for _, t_start, t_end in taken):
            continue
        taken.append((phi_type, start, end))
    taken.sort(key=lambda t: t[1])
    return _merge_adjacent_names(taken)


def _merge_adjacent_names(spans: list[tuple[PhiType, int, int]]) -> list[tuple[PhiType, int, int]]:
    merged: list[tuple[PhiType, int, int]] = []
    for span in spans:
        if merged and span[0] == merged[-1][0] == PhiType.NAME and span[1] - merged[-1][2] == 1:
            merged[-1] = (PhiType.NAME, merged[-1][1], span[2])
        else:
            merged.append(span)
    return merged


def _replacement(phi_type: PhiType, value: str) -> str:
    if phi_type is PhiType.DATE:
        year = _YEAR.search(value)
        return f"[DATE:{year.group(1)}]" if year else "[DATE]"
    if phi_type is PhiType.AGE:
        return "[AGE:90+]"
    return f"[{phi_type.value}]"


def detect_phi(text: str) -> list[PhiSpan]:
    """Return non-overlapping PHI spans found in ``text``, sorted by position."""
    cands: list[tuple[int, PhiType, int, int]] = []
    for rank, (phi_type, pattern, group) in enumerate(_PATTERNS):
        for m in pattern.finditer(text):
            cands.append((rank, phi_type, m.start(group), m.end(group)))
    # Names rank after the structured identifiers so "Dr" street suffixes etc. win.
    name_rank = len(_PATTERNS)
    for phi_type, start, end in _name_spans(text):
        cands.append((name_rank, phi_type, start, end))
    return [
        PhiSpan(
            type=phi_type,
            start=start,
            end=end,
            text=text[start:end],
            replacement=_replacement(phi_type, text[start:end]),
        )
        for phi_type, start, end in _resolve_overlaps(cands)
    ]


def redact(text: str) -> RedactionResult:
    """Replace every detected identifier with a typed placeholder such as ``[NAME]``.

    Dates keep only their year (``[DATE:2091]``) and ages over 89 become ``[AGE:90+]``,
    following the Safe Harbor method. Span offsets refer to the original text.
    """
    spans = detect_phi(text)
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor : span.start])
        parts.append(span.replacement)
        cursor = span.end
    parts.append(text[cursor:])
    return RedactionResult(text="".join(parts), spans=spans)
