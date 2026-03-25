# Process Log - Week 4 of March 2026
This document captures the daily activities, decisions, and reflections during the fourth week of March 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## 2026-03-23 | Final presentation v2 integration + visual enhancement
Intent:
- Consolidate the latest final presentation content into one coherent 15-minute technical flow.
- Increase visual support for explanation-heavy slides while preserving timing discipline.

Action:
- Merged and restructured the final deck into:
	- `research/presentation/Final_presentation_version_2_2026_3_23.md`
- Refined technical mid-section flow (data provenance -> teacher design -> distillation/calibration -> real-time stabilization).
- Added/updated timing allocations to fit strict 15:00 for Slides 1-14.
- Implemented a reusable figure-generation script:
	- `scripts/generate_presentation_figures.py`
- Generated presentation-only PNG assets:
	- `research/presentation/figures/figP1_data_provenance_flow.png`
	- `research/presentation/figures/figP2_teacher_ensemble_arcface.png`
	- `research/presentation/figures/figP3_kd_dkd_decomposition.png`

Result:
- Final deck now has a complete, presentation-ready narrative with stronger graphical support.
- The new figures directly support Slides 6-8 technical explanation and reduce verbal-only burden.

Decision / Interpretation:
- Keep the v2 deck as the primary speaking source for final delivery.
- Preserve generated figures under `research/presentation/figures/` and regenerate via script if style/titles need updates.

Next:
- Align speaking script wording exactly to v2 deck phrasing and timing.
- Keep Slide 15 as Q&A backup only during live delivery.

## 2026-03-24 | New script v2 with explicit math explanation
Intent:
- Produce a new speaking script aligned with the v2 deck and explicitly explain each equation (purpose + project usage).

Action:
- Created/updated:
	- `research/presentation/Final presentation script version 2.md`
- Added slide-by-slide narration for the full 15-minute flow.
- Expanded math explanation coverage for:
	- ArcFace angular-margin term `cos(theta + m)` (teacher separation purpose)
	- KD softening `softmax(z/T)` (dark knowledge transfer)
	- DKD decomposition `L_DKD = alpha*L_TCKD + beta*L_NCKD` (target/non-target decoupling)
	- ECE definition (calibration gap measurement)
	- EMA smoothing equation in real-time inference (jitter reduction mechanism)
- Linked each formula to how it was actually used in this repository workflow.

Result:
- A complete script now exists for `Final_presentation_version_2_2026_3_23.md`, with math made presentation-friendly and evidence-grounded.

Decision / Interpretation:
- This script becomes the main presenter script for final rehearsal.
- Keep the older script as reference only; avoid mixing versions during rehearsal.

Next:
- Run one rehearsal pass at target 14:45-15:00 and compress Slide 11 details if needed.
- Merge this Week-4 entry into `research/process_log/Full process log.md` in the next consolidation cycle.

