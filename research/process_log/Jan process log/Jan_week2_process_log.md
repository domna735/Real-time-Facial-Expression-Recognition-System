# Process Log - Week 2 of January 2026
This document captures the daily activities, decisions, and reflections during the second week of January 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

## 2026-01-05 | NL/NegL Report Consolidation
Intent:
- Consolidate NL/NegL screening evidence into a formal report and make the narrative strictly evidence-backed (offline metrics only).

Action:
- Created/updated the NL/NegL report with Methods → Results → Analysis → Conclusion → Future Work structure.
- Grounded statements in repo artifacts: compare markdowns, per-run reliability metrics, and per-epoch history logs.
- Added mechanism sanity signals (NL/NegL applied fractions) and filled missing gating evidence for KD top-k=0.05.
- Drafted/inserted an evidence-perfect Abstract and Introduction aligned with what was actually measured.

Result:
- Report updated at research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md.
- Key outcomes captured: NL(proto) stable but not consistently better; DKD+NL top-k=0.05 w=0.1 regresses; NegL threshold sensitive; NL+NegL improves raw calibration under DKD in tested config without F1 gains.

Decision / Interpretation:
- Treat results as offline-only evidence (Accuracy/F1/ECE/NLL); real-time stability metrics (flip-rate/jitter) remain a future measurement task.
- Next experiments should prioritize safe regimes (lower NL weight; controlled NegL sweeps) before synergy runs.

Next:
- Run longer-budget confirmations for any promising settings.
- Add demo-log-based stability metrics (flip-rate/confidence stability/confident-wrong) to align evaluation with deployment goals.