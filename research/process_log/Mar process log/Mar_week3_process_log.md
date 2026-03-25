# Process Log - Week 3 of March 2026
This document captures the daily activities, decisions, and reflections during the third week of March 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## 2026-03-17 | Final Report Polish & A4 Print Formatting
Intent:
- Finalize `final report version 3.md` for a formal, physical A4 print submission.
- Expand sparse empirical sections, inject visual figures to improve readability, and convert all unclickable digital artifact references into readable data tables in the appendix.

Action:
- Expanded sparse sections (e.g., student distillation, domain shift, calibration metrics) with detailed academic prose.
- Generated 3 missing quantitative figures (Pipeline Architecture, Data Imbalance, EMA Hysteresis) using Matplotlib.
- Embedded the newly generated figures along with 10 existing figures into the report, standardizing all table numbering and captions.
- Refactored Appendix A.1 by reading raw JSON artifacts (`manifest_validation_all_with_expw.json`, `ensemble_metrics.json`, `reliabilitymetrics.json`, `score_results.json`) via shell and transforming their data into printable Markdown tables.
- Refactored "A.4 Advanced Technical Details" to convert bulleted data into structured tables (Teacher Training Composition, Distillation Split Structure, Per-Source Regression Breakdown, FER2013 Official Protocol).

Result:
- The final report (`final report version 3.md`) is now thoroughly enriched with empirical narratives, perfectly structured data tables, and standard figure captions.
- The Appendix contains human-readable versions of all critical JSON metric artifacts, fully satisfying the constraints of physical A4 evaluation.
- Defended the MobileNetV3 + Hysteresis pipeline conceptually by highlighting its strengths constraint (Expected Calibration Error + Latency trade-offs), proving real-time relevance despite raw F1 benchmark dips.

Decision / Interpretation:
- In physical print submissions, readers cannot click links or inspect JSON files. We must port the "evidence-first" approach directly onto the page via raw data tables.
- The system's edge-optimization and low-latency performance are its primary academic defenses for slightly lower raw-accuracy metrics compared to heavy offline ensembles.

Next:
- Confirm cover page details (Title, Author Name, Date, ID).
- Convert the Final Report Markdown to PDF for official FYP submission.
- Merge/archive the latest process logs into `Full process log.md`.
