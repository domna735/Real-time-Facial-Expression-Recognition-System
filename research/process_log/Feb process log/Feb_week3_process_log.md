# Process Log - Week 3 of February 2026

This document captures daily activities, decisions, and reflections during the third week of February 2026, focusing on reconstructing the FER system under an evidence-first workflow.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## 2026-02-21 | Self-learning+NegL buffer + paper extraction groundwork
Intent:
- Enable a reproducible Self-Learning + Negative Learning (NegL) loop to reduce webcam domain-shift.
- Prepare evidence-backed extraction artifacts from comparison papers so protocol/metric claims can be quoted from on-disk text.

Action:
- Implemented a webcam “self-learn buffer” manifest builder:
	- Script: `scripts/build_webcam_selflearn_buffer.py`
	- Output: `buffer_selflearn/manifest.csv` built from webcam `per_frame.csv` (+ video)
	- Manifest schema includes optional `weight` (for weighted CE) and `neg_label` (explicit NegL target).
- Wired the new manifest fields through the training stack:
	- Dataset parsing + collation support: `src/fer/data/manifest_dataset.py` (optional `weight`, `neg_label`, and `return_meta=True` for safe batch collation)
	- Training consumption: `scripts/train_student.py` (opt-in flags `--manifest-use-weights` and `--manifest-use-neg-label`)
		- Weighted CE implemented as `reduction='none'` + normalized weighted mean.
		- NegL can be driven by manifest-provided `neg_label` when present.
- Extracted text from PDFs under `research/paper compared/` into searchable artifacts:
	- Directory: `outputs/paper_extract/`
	- Files: per-paper `.txt` + `__snippets.md` companions.
	- Attempted structured extraction of AffectNet Table 9; raw page text saved for manual verification:
		- `outputs/paper_extract/affectnet__table9__raw_pages.tsv`
		- `outputs/paper_extract/affectnet__table9__table.tsv` (present but may be unreliable).

- Executed the first end-to-end A/B adaptation attempt on a labeled, recorded webcam session:
	- Baseline checkpoint: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
	- Session: `demo/outputs/20260126_205446/` (contains `per_frame.csv`, `events.csv`, `session_annotated.mp4`, and `score_results.json`)
	- Buffer built at: `demo/outputs/20260126_205446/buffer_selflearn/manifest.csv` + `buffer_summary.json`
	- Initial fine-tune candidates (head-only / BN-only / lastblock_head fallback) were evaluated against the offline gate.

- Diagnosed two root causes behind “adaptation regressions”:
	- **Preprocessing mismatch:** baseline checkpoint stored `use_clahe=True`, while early adaptation checkpoints had `use_clahe=False`, making eval-only comparisons not apples-to-apples.
	- **BatchNorm running-stat drift:** even under head-only tuning, `model.train()` updates BN running mean/var on a tiny buffer, causing large distribution shifts.

- Implemented fixes to support fair A/B scoring and safer adaptation:
	- Patched `scripts/train_student.py` so that when `--tune` is not `all`, BatchNorm layers are forced into eval mode during training (freezing running stats).
	- Added `scripts/reinfer_webcam_session.py` to re-run inference on the recorded session video while **preserving** `manual_label` and `time_sec` from an existing `per_frame.csv` (enables fair A/B re-scoring without re-labeling).

Result:
- Self-learning data path exists end-to-end (webcam logs → buffer manifest → dataset → weighted CE + NegL in training).
- Paper text is now greppable/searchable locally even when snippet heuristics miss (searching may require enabling search in excluded folders).

- Offline gate (eval-only manifest) became stable once preprocessing and BN behavior were controlled:
	- Adapted checkpoint (gate-passing): `outputs/students/DA/mnv3_webcamselflearn_negl_clahe_head_frozebn_20260221_211025/best.pt`
	- Gate artifacts:
		- Baseline: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/baseline/`
		- Adapted: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/adapted_clahe_head_frozebn/`

- Deployment-facing A/B on the *same* labeled session regressed despite passing the gate:
	- Baseline score: `demo/outputs/20260126_205446/score_results.json`
	- Adapted score: `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`
	- Smoothed metrics (same scored_frames=4154):
		- Baseline: accuracy 0.5879, macro-F1 0.5248, minority-F1(lowest-3) 0.1609, jitter 14.86 flips/min
		- Adapted: accuracy 0.5269, macro-F1 0.4667, minority-F1(lowest-3) 0.1384, jitter 14.16 flips/min

Decision / Interpretation:
- Treat webcam pseudo-label adaptation as a controlled, opt-in fine-tune step (manifest-driven), gated by (a) offline eval-only checks and (b) webcam scoring stability metrics.
- For paper comparison, use extracted `.txt` artifacts as the canonical quoting source; do not trust automatic “snippet” extraction to be complete.

- For adaptation: passing the offline gate is necessary but not sufficient; A/B on the same labeled session remains the decisive deployment-facing check.
- Ensure adaptation candidates match baseline preprocessing (e.g., CLAHE) and avoid BN running-stat drift; otherwise “regressions” may be measurement artifacts rather than true model degradation.

Next:
- Build an evidence-backed comparison matrix (per paper): dataset(s) used, split protocol, label space, metric(s), and whether comparable to our FER2013 official split evaluation.
- For fast extraction, search `outputs/paper_extract/*.txt` with “include ignored files” enabled (the folder may be excluded by editor search settings).

- Iterate the webcam adaptation loop by changing **one knob at a time** (buffer thresholds, ratio of positive pseudo-labels vs NegL-only, learning rate / tune policy), and re-run: (1) eval-only gate, then (2) identical-session A/B scoring.

## 2026-02-25 | Academic audit + consistency fixes (final report + domain-shift docs)
Intent:
- Do a full academic-quality cross-check of the final report against supporting plan/report docs and process logs.
- Ensure negative/fail results (especially webcam self-learning + NegL domain-shift A/B) are explained correctly, without over-claiming.
- Remove terminology ambiguity (NL vs NegL) and align wording/timeline across documents.

Action:
- Audited the main final report for internal consistency (methods, results, timeline, conclusions), with focus on:
	- Domain-shift/self-learning+NegL method description and its Feb-21 controlled A/B evidence.
	- Correct interpretation of “passed offline gate but regressed on same-session webcam A/B”.
	- Avoiding protocol mismatch explanations being lost (CLAHE mismatch, BN running-stat drift).
- Cross-checked the final report narrative against:
	- Domain-shift plan acceptance criteria: `research/domain shift improvement via Self-Learning + Negative Learning plan/04_metrics_acceptance.md`
	- NL/NegL plan acceptance criteria: `research/nl_negl_plan/04_metrics_acceptance.md`
	- The Feb-week3 log entry (2026-02-21) for artifact paths + metric values.
- Inspected the self-learning buffer policy implementation to ensure the report’s explanation matches the code:
	- Script checked: `scripts/build_webcam_selflearn_buffer.py` (confirmed medium-confidence frames are NegL-only with `weight=0` and `neg_label=predicted label`).
- Patched documentation to remove contradictions and improve academic clarity:
	- Final report: `research/final report/final report version 2.md`
	- Domain-shift report addendum/status alignment:
		`research/domain shift improvement via Self-Learning + Negative Learning plan/domain shift improvement via Self-Learning + Negative Learning report/domain shift improvement via Self-Learning + Negative Learning report.md`
	- NL/NegL report terminology clarification:
		`research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`

Result:
- The final report now describes the Feb-21 webcam self-learning + NegL attempt with clearer method detail (buffer source, confidence bands, how `weight` and `neg_label` are used) and explicitly frames the outcome as a negative result under a controlled A/B protocol.
- Terminology is disambiguated so academic readers do not confuse “NL (Nested Learning)” with “NegL (negative/complementary-label learning)”.
- Supporting reports no longer contradict the final report (they acknowledge the attempted Feb-21 run and its negative A/B result).

Decision / Interpretation:
- Treat the Feb-21 webcam adaptation result as evidence of a failure mode (offline gate is necessary but not sufficient), not evidence that self-learning/NegL can never work.
- Keep domain-shift updates evidence-first: every claim should point to a reproducible artifact path (gate dirs, checkpoints, session score JSON) and match the implemented pipeline.

Next:
- Optional: do a second-pass “academic polish only” sweep on `final report version 2.md` (tense consistency, citation/reference formatting, small phrasing improvements) without changing any reported numbers or artifact paths.