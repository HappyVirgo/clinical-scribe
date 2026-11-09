# Evaluation results

> Offline baseline only (rule-based note provider + BM25 coder). No LLM was called.
> Synthetic data. NOT FOR CLINICAL USE.

- Command: `python -m eval.run --repeats 20`
- Data: 28 synthetic transcripts in `eval/data/` (17 dev / 11 held-out test), 360 gold-labelled sentences, 133 gold PHI spans
- Machine: 12th Gen Intel(R) Core(TM) i7-12700; Windows 11; Python 3.13.12
- Date: 2026-11-06

## Summary

| Metric | dev | test | all |
|---|---|---|---|
| PHI recall, type-matched (micro) | 0.99 (82/83) | 0.80 (40/50) | 0.92 (122/133) |
| PHI recall, any type redacted | 0.99 (82/83) | 0.80 (40/50) | 0.92 (122/133) |
| PHI precision (micro) | 1.00 (82/82) | 1.00 (40/40) | 1.00 (122/122) |
| Sentence -> section accuracy | 0.96 (218/226) | 0.88 (118/134) | 0.93 (336/360) |
| ICD-10 top-1 (end-to-end) | 0.94 | 0.73 | 0.86 |
| ICD-10 top-3 (end-to-end) | 0.94 | 0.91 | 0.93 |
| ICD-10 recall@3 (end-to-end) | 0.89 | 0.91 | 0.90 |
| ICD-10 top-1 (gold Assessment) | 1.00 | 0.82 | 0.93 |
| ICD-10 top-3 (gold Assessment) | 1.00 | 1.00 | 1.00 |

## PHI by type

| Type | split | gold | recall | precision |
|---|---|---|---|---|
| ACCOUNT | dev | 3 | 1.00 (3/3) | 1.00 (3/3) |
| ACCOUNT | test | 1 | 1.00 (1/1) | 1.00 (1/1) |
| ADDRESS | dev | 2 | 1.00 (2/2) | 1.00 (2/2) |
| ADDRESS | test | 1 | 1.00 (1/1) | 1.00 (1/1) |
| AGE | dev | 3 | 0.67 (2/3) | 1.00 (2/2) |
| AGE | test | 1 | 0.00 (0/1) | n/a |
| DATE | dev | 16 | 1.00 (16/16) | 1.00 (16/16) |
| DATE | test | 9 | 0.89 (8/9) | 1.00 (8/8) |
| EMAIL | dev | 1 | 1.00 (1/1) | 1.00 (1/1) |
| EMAIL | test | 3 | 1.00 (3/3) | 1.00 (3/3) |
| FAX | dev | 1 | 1.00 (1/1) | 1.00 (1/1) |
| FAX | test | 2 | 1.00 (2/2) | 1.00 (2/2) |
| IP | dev | 1 | 1.00 (1/1) | 1.00 (1/1) |
| IP | test | 0 | n/a | n/a |
| MRN | dev | 7 | 1.00 (7/7) | 1.00 (7/7) |
| MRN | test | 3 | 0.67 (2/3) | 1.00 (2/2) |
| NAME | dev | 42 | 1.00 (42/42) | 1.00 (42/42) |
| NAME | test | 27 | 0.78 (21/27) | 1.00 (21/21) |
| PHONE | dev | 3 | 1.00 (3/3) | 1.00 (3/3) |
| PHONE | test | 3 | 0.67 (2/3) | 1.00 (2/2) |
| SSN | dev | 1 | 1.00 (1/1) | 1.00 (1/1) |
| SSN | test | 0 | n/a | n/a |
| URL | dev | 1 | 1.00 (1/1) | 1.00 (1/1) |
| URL | test | 0 | n/a | n/a |
| ZIP | dev | 2 | 1.00 (2/2) | 1.00 (2/2) |
| ZIP | test | 0 | n/a | n/a |

## Section confusion (all splits; rows = gold, columns = predicted)

| gold \ pred | S | O | A | P | X |
|---|---|---|---|---|---|
| S | 77 | 0 | 0 | 0 | 2 |
| O | 0 | 65 | 0 | 0 | 4 |
| A | 0 | 1 | 26 | 0 | 1 |
| P | 2 | 0 | 1 | 98 | 2 |
| X | 3 | 4 | 0 | 4 | 70 |

## Latency per note (offline, 560 runs = 28 notes x 20)

| Stage | p50 ms | p95 ms |
|---|---|---|
| phi | 0.615 | 0.877 |
| note | 0.394 | 0.583 |
| coding | 0.144 | 0.257 |
| total | 1.157 | 1.646 |
| wall | 1.167 | 1.657 |
