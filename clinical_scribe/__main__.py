"""Command line interface: ``python -m clinical_scribe note transcript.txt``."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from clinical_scribe.notes import NoteGenerationError, RuleBasedNoteProvider, SOAPNote
from clinical_scribe.pipeline import DISCLAIMER, ScribePipeline, ScribeResult


def _render(result: ScribeResult) -> str:
    note: SOAPNote = result.note
    lines = [f"# SOAP note draft (provider: {result.provider})", f"> {DISCLAIMER}", ""]
    for title, items in (
        ("Subjective", note.subjective),
        ("Objective", note.objective),
        ("Assessment", note.assessment),
        ("Plan", note.plan),
    ):
        lines.append(f"## {title}")
        lines.extend(f"- {item}" for item in items or ["(none captured)"])
        lines.append("")
    lines.append("## Medications")
    for med in note.medications or []:
        details = ", ".join(x for x in (med.dose, med.frequency) if x)
        lines.append(f"- {med.name}" + (f" ({details})" if details else ""))
    if not note.medications:
        lines.append("- (none captured)")
    lines += ["", f"## Follow-up\n{note.follow_up or '(none captured)'}", ""]
    lines.append("## Suggested ICD-10-CM codes (decision support only)")
    for code in result.suggested_codes:
        lines.append(f"- {code.code} {code.description} (score {code.score}; matched: {', '.join(code.evidence)})")
    counts = Counter(span.type.value for span in result.phi_spans)
    phi = ", ".join(f"{name} x{n}" for name, n in sorted(counts.items()))
    lines += ["", f"PHI redacted: {phi or 'none'}", f"Timings (ms): {result.timings_ms}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="clinical_scribe", description=DISCLAIMER)
    sub = parser.add_subparsers(dest="command", required=True)
    note_cmd = sub.add_parser("note", help="draft a SOAP note from a transcript file")
    note_cmd.add_argument("path", type=Path)
    note_cmd.add_argument("--json", action="store_true", help="print the full JSON result")
    note_cmd.add_argument("--offline", action="store_true", help="force the rule-based provider")
    args = parser.parse_args(argv)

    transcript = args.path.read_text(encoding="utf-8")
    pipeline = ScribePipeline(provider=RuleBasedNoteProvider() if args.offline else None)
    try:
        result = pipeline.run(transcript)
    except NoteGenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2) if args.json else _render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
