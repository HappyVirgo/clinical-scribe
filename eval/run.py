"""Evaluate the offline pipeline on the synthetic gold set and write ``eval/RESULTS.md``.

Usage: ``python -m eval.run [--repeats N] [--out PATH]``

Reports, for the dev split (used while writing heuristics), the held-out test split and both:
PHI recall/precision per type, ICD-10 top-1/top-3 accuracy, sentence-to-section accuracy of
the rule-based provider, and p50/p95 end-to-end latency per note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from clinical_scribe.notes import RuleBasedNoteProvider, Section, classify_sentence
from clinical_scribe.phi import PhiSpan, detect_phi, redact
from clinical_scribe.pipeline import ScribePipeline, default_suggester
from clinical_scribe.text import Sentence
from eval.dataset import Case, GoldSpan, load_cases

SPLITS = ("dev", "test", "all")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def phi_counts(cases: Sequence[Case]) -> dict[str, Counter[str]]:
    """Per-type gold, recalled, predicted and correct-prediction counts."""
    counts: dict[str, Counter[str]] = {k: Counter() for k in ("gold", "hit", "hit_any", "pred", "pred_ok")}
    for case in cases:
        pred: list[PhiSpan] = detect_phi(case.transcript)
        gold: list[GoldSpan] = case.phi
        for g in gold:
            counts["gold"][g.type] += 1
            same = [p for p in pred if _overlaps(g.start, g.end, p.start, p.end)]
            counts["hit_any"][g.type] += bool(same)
            counts["hit"][g.type] += any(p.type.value == g.type for p in same)
        for p in pred:
            counts["pred"][p.type.value] += 1
            counts["pred_ok"][p.type.value] += any(
                g.type == p.type.value and _overlaps(g.start, g.end, p.start, p.end) for g in gold
            )
    return counts


def section_scores(cases: Sequence[Case]) -> tuple[int, int, Counter[tuple[str, str]]]:
    """Correct count, total and (gold, predicted) confusion counts over gold sentences."""
    confusion: Counter[tuple[str, str]] = Counter()
    for case in cases:
        for s in case.sentences:
            predicted = classify_sentence(Sentence(s.role, redact(s.text).text))
            confusion[(s.section.value, predicted.value)] += 1
    correct = sum(n for (g, p), n in confusion.items() if g == p)
    return correct, sum(confusion.values()), confusion


def coding_scores(cases: Sequence[Case], assessments: dict[str, str]) -> tuple[float, float, float]:
    """Top-1 accuracy, top-3 accuracy (any gold code) and recall@3 over gold codes."""
    suggester = default_suggester()
    top1 = top3 = 0
    recalled = total = 0
    for case in cases:
        ranked = [c.code for c in suggester.suggest(assessments[case.case_id], k=3)]
        gold = set(case.codes)
        top1 += bool(ranked) and ranked[0] in gold
        top3 += bool(gold & set(ranked))
        recalled += len(gold & set(ranked))
        total += len(gold)
    n = max(len(cases), 1)
    return top1 / n, top3 / n, recalled / max(total, 1)


def latency(cases: Sequence[Case], repeats: int) -> dict[str, list[float]]:
    """Per-stage wall-clock samples (ms) of the full offline pipeline."""
    pipeline = ScribePipeline(provider=RuleBasedNoteProvider())
    for case in cases:  # warm-up: regex compilation, BM25 index
        pipeline.run(case.transcript)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        for case in cases:
            start = time.perf_counter()
            result = pipeline.run(case.transcript)
            samples.setdefault("wall", []).append((time.perf_counter() - start) * 1000)
            for stage, ms in result.timings_ms.items():
                samples.setdefault(stage, []).append(ms)
    return samples


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _ratio(num: int, den: int) -> str:
    return f"{num / den:.2f} ({num}/{den})" if den else "n/a"


def _machine() -> str:
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def build_report(cases: list[Case], repeats: int) -> str:
    """Run every metric and render the markdown report."""
    by_split = {
        "dev": [c for c in cases if c.split == "dev"],
        "test": [c for c in cases if c.split == "test"],
        "all": cases,
    }
    pipeline = ScribePipeline(provider=RuleBasedNoteProvider())
    e2e_assessment = {c.case_id: " ".join(pipeline.run(c.transcript).note.assessment) for c in cases}
    gold_assessment = {
        c.case_id: " ".join(s.text for s in c.sentences if s.section is Section.ASSESSMENT) for c in cases
    }
    n_sent = sum(len(c.sentences) for c in cases)
    out = [
        "# Evaluation results",
        "",
        "> Offline baseline only (rule-based note provider + BM25 coder). No LLM was called.",
        "> Synthetic data. NOT FOR CLINICAL USE.",
        "",
        f"- Command: `python -m eval.run --repeats {repeats}`",
        f"- Data: {len(cases)} synthetic transcripts in `eval/data/` "
        f"({len(by_split['dev'])} dev / {len(by_split['test'])} held-out test), "
        f"{n_sent} gold-labelled sentences, {sum(len(c.phi) for c in cases)} gold PHI spans",
        f"- Machine: {_machine()}",
        f"- Date: {dt.date.today().isoformat()}",
        "",
        "## Summary",
        "",
        "| Metric | dev | test | all |",
        "|---|---|---|---|",
    ]
    phi = {s: phi_counts(by_split[s]) for s in SPLITS}
    sec = {s: section_scores(by_split[s]) for s in SPLITS}
    e2e = {s: coding_scores(by_split[s], e2e_assessment) for s in SPLITS}
    oracle = {s: coding_scores(by_split[s], gold_assessment) for s in SPLITS}

    def row(label: str, cells: dict[str, str]) -> None:
        out.append(f"| {label} | " + " | ".join(cells[s] for s in SPLITS) + " |")

    def total(s: str, key: str) -> int:
        return sum(phi[s][key].values())

    row("PHI recall, type-matched (micro)", {s: _ratio(total(s, "hit"), total(s, "gold")) for s in SPLITS})
    row("PHI recall, any type redacted", {s: _ratio(total(s, "hit_any"), total(s, "gold")) for s in SPLITS})
    row("PHI precision (micro)", {s: _ratio(total(s, "pred_ok"), total(s, "pred")) for s in SPLITS})
    row("Sentence -> section accuracy", {s: _ratio(sec[s][0], sec[s][1]) for s in SPLITS})
    row("ICD-10 top-1 (end-to-end)", {s: f"{e2e[s][0]:.2f}" for s in SPLITS})
    row("ICD-10 top-3 (end-to-end)", {s: f"{e2e[s][1]:.2f}" for s in SPLITS})
    row("ICD-10 recall@3 (end-to-end)", {s: f"{e2e[s][2]:.2f}" for s in SPLITS})
    row("ICD-10 top-1 (gold Assessment)", {s: f"{oracle[s][0]:.2f}" for s in SPLITS})
    row("ICD-10 top-3 (gold Assessment)", {s: f"{oracle[s][1]:.2f}" for s in SPLITS})

    out += ["", "## PHI by type", "", "| Type | split | gold | recall | precision |", "|---|---|---|---|---|"]
    types = sorted(set(phi["all"]["gold"]) | set(phi["all"]["pred"]))
    for t in types:
        for s in ("dev", "test"):
            c = phi[s]
            out.append(
                f"| {t} | {s} | {c['gold'][t]} | {_ratio(c['hit'][t], c['gold'][t])} "
                f"| {_ratio(c['pred_ok'][t], c['pred'][t])} |"
            )

    labels = [x.value for x in Section]
    out += ["", "## Section confusion (all splits; rows = gold, columns = predicted)", ""]
    out.append("| gold \\ pred | " + " | ".join(labels) + " |")
    out.append("|---" * (len(labels) + 1) + "|")
    for g in labels:
        out.append(f"| {g} | " + " | ".join(str(sec["all"][2][(g, p)]) for p in labels) + " |")

    samples = latency(cases, repeats)
    out += [
        "",
        f"## Latency per note (offline, {len(samples['wall'])} runs = {len(cases)} notes x {repeats})",
        "",
        "| Stage | p50 ms | p95 ms |",
        "|---|---|---|",
    ]
    for stage in ("phi", "note", "coding", "total", "wall"):
        values = samples[stage]
        out.append(f"| {stage} | {_fmt(statistics.median(values))} | {_fmt(_pct(values, 0.95))} |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation, print the report and write it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20, help="latency repetitions per note")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "RESULTS.md")
    args = parser.parse_args(argv)
    report = build_report(load_cases(), args.repeats)
    print(report)
    args.out.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
