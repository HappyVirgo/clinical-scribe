"""Tests for PHI detection and redaction (synthetic identifiers only)."""

import pytest

from clinical_scribe.phi import PhiType, detect_phi, redact


def _types(text: str) -> list[tuple[PhiType, str]]:
    return [(s.type, s.text) for s in detect_phi(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Call 555-010-1234 today.", (PhiType.PHONE, "555-010-1234")),
        ("Call (555) 010-1234 today.", (PhiType.PHONE, "(555) 010-1234")),
        ("Send the fax to 555.010.9876 please.", (PhiType.FAX, "555.010.9876")),
        ("Email me at z.q@example.com now.", (PhiType.EMAIL, "z.q@example.com")),
        ("SSN 000-12-3456 on file.", (PhiType.SSN, "000-12-3456")),
        ("MRN: ZZ-00412 noted.", (PhiType.MRN, "ZZ-00412")),
        ("member ID HMO-4401928 active.", (PhiType.ACCOUNT, "HMO-4401928")),
        ("Portal https://portal.example.test/u/1 works.", (PhiType.URL, "https://portal.example.test/u/1")),
        ("Login from 10.0.0.7 today.", (PhiType.IP, "10.0.0.7")),
        ("She is a 94-year-old woman.", (PhiType.AGE, "94-year-old")),
    ],
)
def test_structured_identifiers(text: str, expected: tuple[PhiType, str]) -> None:
    assert expected in _types(text)


@pytest.mark.parametrize(
    "date",
    ["03/14/2091", "2092-02-03", "March 3rd, 2093", "17 May 2092", "Jan 9, 2094", "October of 2091"],
)
def test_date_formats_keep_year(date: str) -> None:
    result = redact(f"Seen on {date} in clinic.")
    assert result.spans[0].type is PhiType.DATE
    assert "[DATE:20" in result.text
    assert date not in result.text


def test_address_and_zip() -> None:
    found = _types("Lives at 742 Fakewood Lane, Nowhereville, ZZ 00042.")
    assert (PhiType.ADDRESS, "742 Fakewood Lane, Nowhereville") in found
    assert (PhiType.ZIP, "00042") in found


def test_names_from_titles_relatives_and_signatures() -> None:
    text = (
        "Doctor: Good morning Mr. Quibbleworth.\n"
        "Patient: My wife Glorbina drove me.\n"
        "Electronically signed: Fennimore Wobblesprocket, MD"
    )
    names = {t for kind, t in _types(text) if kind is PhiType.NAME}
    assert names == {"Quibbleworth", "Glorbina", "Fennimore Wobblesprocket"}


def test_name_propagates_to_uncued_mentions() -> None:
    text = "Patient name: Zorblat Quibbleworth\nDoctor: Zorblat, how is the knee?"
    spans = [s for s in detect_phi(text) if s.type is PhiType.NAME]
    assert [s.text for s in spans] == ["Zorblat Quibbleworth", "Zorblat"]


def test_header_line_names_without_cue() -> None:
    found = _types("GI clinic, Barnaby Crumblewick, seen 6/21/2092")
    assert (PhiType.NAME, "Barnaby Crumblewick") in found


def test_speaker_labels_are_not_names() -> None:
    text = "Patient: Sure, that works.\nDoctor: Great, Thanks."
    assert [s for s in detect_phi(text) if s.type is PhiType.NAME] == []


def test_clinical_numbers_are_not_phi() -> None:
    text = "Doctor: BP is 142/90, pulse 88, A1c 7.2, take 500 mg twice a day for 10 days, age 45."
    assert detect_phi(text) == []


def test_young_age_not_redacted() -> None:
    assert detect_phi("A 45-year-old man.") == []


def test_redaction_offsets_refer_to_original_text() -> None:
    text = "Dr. Wobblesprocket called 555-010-1234."
    result = redact(text)
    for span in result.spans:
        assert text[span.start : span.end] == span.text
    assert result.text == "Dr. [NAME] called [PHONE]."


def test_two_digit_year_redacts_without_year() -> None:
    assert redact("DOB 3/4/91.").text == "DOB [DATE]."


def test_empty_text() -> None:
    result = redact("")
    assert result.text == "" and result.spans == []
