# NL / NegL Plan (Student Training) — Index

Target reference notes (old report PDF):
- `research/old report/KD and DKD Problems and NL and NegL Solutions Personal Discussion Notes.pdf`

This folder is a *do-one-step-next* plan for continuing student training beyond the current CE→KD→DKD baseline.

## What we already have (in code)
- NL scaffolding (memory gate): `src/fer/nl/memory.py`
- NegL scaffolding (complementary-label loss): `src/fer/negl/losses.py`
- NL smoke test: `scripts/smoke_nl.py`

## Plan documents
- 01_problem_map.md — KD/DKD weaknesses → NL/NegL opportunities (table + deployment signals)
- 02_experiment_framework.md — hypotheses, steps, and evidence checklist
- 03_implementation_steps.md — the exact engineering tasks (flags, wiring, logging)
- 04_metrics_acceptance.md — metrics definitions + pass/fail gates (Macro-F1, Minority-F1, ECE, NLL, flip-rate)

## Quick next action
Start with:
1) Read 01_problem_map.md (align with your supervisor narrative)
2) Run NL smoke test (`scripts/smoke_nl.py`) to confirm the scaffolding is alive
3) Implement NegL wiring first (lower risk than NL meta-optimizer style changes)
