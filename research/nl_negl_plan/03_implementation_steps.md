# Part 3 — Implementation Steps (Engineering)

Goal: wire NegL and NL into *student training* with minimal risk, stable Windows behavior, and good logging.

## Phase A — Baseline lock (no new methods)
1) Confirm baseline commands still run.
2) Confirm metrics are emitted (history/reliability/calibration).
3) Confirm evaluation protocol is consistent (`Training_data_cleaned/test_all_sources.csv`, RAF-only CSVs as secondary).

## Phase B — NegL wiring (recommended first)
Why first: NegL is an additive loss and easier to ablate than NL optimizer/memory changes.

Tasks:
1) Add CLI flags to `scripts/train_student.py`:
   - `--use-negl`
   - `--negl-weight` (λ)
   - `--negl-ratio` (fraction of minibatch to apply)
   - `--negl-gate` (e.g., `none|entropy|consistency`)
2) Implement complementary-label sampling:
   - Default: uniform random wrong class.
   - Later: teacher-guided sampling (confusion-based).
3) Add uncertainty gating:
   - Entropy gate: apply NegL only if $H(p)$ above threshold.
4) Logging:
   - Report NegL loss value, applied ratio, entropy stats.

## Phase C — NL wiring (memory / gating)
Current code status:
- `src/fer/nl/memory.py` provides `AssociativeMemory` + `apply_memory_gate` (scaffold).
- `scripts/smoke_nl.py` confirms it runs.

Tasks:
1) Add CLI flags:
   - `--use-nl`
   - `--nl-hidden-dim`, `--nl-layers`
   - `--nl-gate-target` (e.g., `grads|loss_weights`)
2) Choose minimal safe integration point:
   - Option 1: gate gradient magnitude (like the smoke test) for selected parameters.
   - Option 2: gate per-sample loss weights (safer, no touching optimizer grads directly).
3) Logging:
   - Gate statistics (mean/min/max), correlation with minority classes.

## Phase D — Combined pipeline
- Run only after Phase B and C individually pass their acceptance criteria.

## Phase E — Real-time signals
- Add a small “flip-rate” summarizer for demo CSV logs (separate script) only if needed.
