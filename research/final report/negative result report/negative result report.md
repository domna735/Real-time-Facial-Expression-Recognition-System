# Negative Result Report (Academic)

Project: Real-time Facial Expression Recognition System (FER)  
Document date: 2026-02-25  
Scope period: Aug 2025 – Feb 2026  
Companion document: `research/final report/final report version 2.md`

---

## Abstract

This report consolidates the project’s key negative outcomes into a single evidence-first, academically interpretable record. The negative results are not treated as side findings; they are central to understanding why deployment-facing FER remains difficult under domain shift and protocol mismatch. Across offline hard gates (`eval_only`, `expw_full`, mixed-source tests), online webcam replay A/B, and NL/NegL auxiliary-loss screening, we observe a repeated pattern: improvements in one objective (e.g., calibration) do not reliably transfer to another objective (e.g., macro-F1, minority-F1, or real-time stability). In particular, a conservative 2026-02-21 Self-Learning + manifest-driven NegL adaptation candidate passed the offline safety gate within rounding but regressed on identical-session webcam replay metrics. This establishes a core methodological conclusion: offline non-regression is necessary but insufficient for deployment-facing gain claims. The report distinguishes evidence-backed findings from hypotheses, lists likely mechanisms, identifies threats to validity, and defines pre-registered next-step experiments.

---

## 1. Purpose and Contribution of Negative Results

The purpose of this document is to answer three research-critical questions:

1. Which negative results are reproducibly supported by stored artifacts?
2. Do these negative results have credible explanations (or only speculation)?
3. What concrete next experiments should be run to convert negative findings into methodological progress?

In this project, negative results are scientifically useful because they:

- Expose failure modes hidden by aggregate metrics.
- Prevent over-claiming from protocol-mismatched comparisons.
- Justify strict safety-gated adaptation for deployment-facing FER.
- Provide a traceable basis for supervisor-facing analytical comparison.

---

## 2. Evidence Standard and Interpretation Rules

### 2.1 Evidence standard used in this report

A result is treated as evidence-backed only if:

1. Baseline and candidate are scored under the same protocol definition.
2. Artifact pairs exist for both sides (JSON/CSV/compare markdown).
3. Confounds are controlled or explicitly acknowledged.

### 2.2 Interpretation labels used throughout

- **Evidence-backed finding:** directly supported by stored artifacts.
- **Hypothesis:** plausible mechanism requiring further one-knob validation.

### 2.3 Primary evidence sources

- Final integrated report: `research/final report/final report version 2.md`
- Dataset diagnostics: `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`
- Teacher hard-gate interpretation:
  - `research/issue__teacher_hard_gates__20260209.md`
  - `research/issue__teacher_metrics_interpretation__20260209.md`
- NL/NegL study: `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`
- Domain-shift adaptation report:
  - `research/domain shift improvement via Self-Learning + Negative Learning plan/domain shift improvement via Self-Learning + Negative Learning report/domain shift improvement via Self-Learning + Negative Learning report.md`

---

## 3. Negative Result Matrix (Executive Index)

| ID | Negative result statement | Status | Main evidence |
| --- | --- | --- | --- |
| NR-1 | 2026-02-21 Self-Learning + manifest-driven NegL candidate passed offline eval-only non-regression (within rounding) but regressed on identical-session webcam replay | Evidence-backed | `demo/outputs/20260126_205446/score_results.json`, `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`, `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/`, `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/` |
| NR-2 | Early head-only and BN-only adaptation candidates failed offline eval-only gate (macro-F1 drop) | Evidence-backed | `outputs/evals/students/_baseline_CE20251223_eval_only_test/reliabilitymetrics.json`, `outputs/evals/students/FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358/reliabilitymetrics.json`, `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/reliabilitymetrics.json` |
| NR-3 | NL/NegL screening under short-budget KD/DKD did not deliver consistent macro-F1/minority-F1 improvements; several settings regressed | Evidence-backed | `outputs/students/_compare*.md`, run `history.json`, `reliabilitymetrics.json` summarized in `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md` |
| NR-4 | KD/DKD improved calibration (TS ECE/TS NLL) but did not outperform CE macro-F1 in main HQ-train snapshot | Evidence-backed | Student `reliabilitymetrics.json` artifacts listed in final report Section 4.4 |
| NR-5 | High Stage-A teacher validation (~0.78–0.79 macro-F1) did not transfer to hard-gate robustness (`eval_only`, `expw_full`) | Evidence-backed | `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`, issue notes above |
| NR-6 | Poor hard-gate performance concentrates in minority/confusable classes (especially Fear/Disgust); calibration fixes alone do not solve macro-F1 | Evidence-backed | `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`, offline suite artifacts |
| NR-7 | LP-loss short-budget addendum improved some calibration terms but did not produce a clear ExpW macro-F1 gain | Evidence-backed | final report Section 9.3.4 and post-eval artifacts (`outputs/evals/students/*20260205*/reliabilitymetrics.json`) |

---

## 4. Detailed Negative Results

## 4.1 NR-1: Offline gate pass, webcam replay regression (critical negative result)

### 4.1.1 Experimental intent

The adaptation loop was designed to improve target-domain webcam behavior while avoiding broad-distribution regression:

- Build a session-specific self-learning buffer.
- Apply conservative fine-tuning with confidence-banded supervision.
- Promote only if offline non-regression and deployment-facing replay improve.

### 4.1.2 Protocol controls (fairness conditions)

- Same recorded session and same manual labels.
- Preprocessing harmonized (`use_clahe` mismatch corrected).
- BatchNorm running-stat updates prevented during non-`all` tuning in the adaptation pass.

### 4.1.3 Outcome (evidence-backed)

Smoothed metrics on the same scored frames (`n=4154`):

- Baseline: accuracy `0.5879`, macro-F1 `0.5248`, minority-F1(lowest-3) `0.1609`, jitter `14.86` flips/min.
- Adapted: accuracy `0.5269`, macro-F1 `0.4667`, minority-F1(lowest-3) `0.1384`, jitter `14.16` flips/min.

Interpretation:

- The adapted checkpoint slightly reduced jitter but materially regressed deployment-facing accuracy and macro-F1.
- Therefore, adaptation is a fail for this session under the pre-defined webcam objective.

### 4.1.4 Explanations

**Evidence-backed explanation:**

- Offline non-regression and deployment replay are different objectives; satisfying one does not guarantee the other.

**Hypotheses requiring direct ablation:**

1. Narrow, correlated session buffer induced localized overfit.
2. Transition-frame pseudo-label noise shifted class boundaries.
3. Medium-confidence NegL policy (`neg_label=<predicted_label>`) may suppress correct-but-uncertain classes.
4. Probability-margin changes degraded EMA/hysteresis behavior without large offline static-loss signals.

### 4.1.5 Threats to validity

- Single-session evidence cannot generalize to all users/cameras.
- Session emotion composition may influence replay outcomes.

### 4.1.6 Immediate next experiments

1. One-knob ablation: positives-only vs NegL-only vs combined.
2. Policy ablation: medium-confidence ignore vs complementary negative target vs current rule.
3. Multi-session replication with fixed scoring policy.

---

## 4.2 NR-2: Early head-only and BN-only adaptation failed eval-only safety gate

### 4.2.1 Outcome (evidence-backed)

Against baseline (`raw macro-F1=0.4859` on eval-only):

- Head-only FT: `0.4508` macro-F1 (fail).
- BN-only FT: `0.4513` macro-F1 (fail).

### 4.2.2 Explanation status

**Evidence-backed:** adaptation can regress broad-distribution macro-F1 even under seemingly conservative update scopes.

**Hypotheses:**

- Target buffer narrowness and correlated samples.
- BN dynamics and small-buffer statistics sensitivity.
- Update scale (LR/epoch) too strong for pseudo-label reliability.

### 4.2.3 Research implication

This negative result justifies the project’s strict “gate-first, promote-later” discipline.

---

## 4.3 NR-3: NL/NegL screening did not provide consistent offline gains

### 4.3.1 Outcome (evidence-backed)

Across KD/DKD screening comparisons:

- No robust, repeatable macro-F1/minority-F1 lift over baseline.
- Some runs improved selected calibration terms while harming macro-F1.
- At least one NL-only DKD configuration showed strong regression.

Mechanism signals from `history.json`:

- High-threshold NegL can become too sparse (`applied_frac` very low).
- Threshold-based NL can decay toward inactivity.
- Top-k NL keeps activity but can over-regularize at tested weights.

### 4.3.2 Explanation status

**Evidence-backed:** gating behavior (inactive vs overactive) is a first-order determinant of outcome.

**Hypotheses:**

- Auxiliary losses conflict with teacher-induced structure in uncertain regions.
- Imbalance/noise amplifies minority-class harm under negative-style penalties.
- Short-budget schedules underrepresent potential long-horizon stabilization.

### 4.3.3 Research implication

NL/NegL cannot be claimed as drop-in improvements in current tuning regimes; they remain conditional research components.

---

## 4.4 NR-4: KD/DKD improved calibration but not CE macro-F1 in main snapshot

### 4.4.1 Outcome (evidence-backed)

In the Dec-2025 HQ-train evaluation snapshot:

- CE had the best raw macro-F1.
- KD/DKD had better temperature-scaled calibration metrics.

### 4.4.2 Interpretation

This is a negative result only if the objective is “macro-F1 improvement over CE.” Under that criterion, KD/DKD failed in this snapshot.

### 4.4.3 Explanation status

**Evidence-backed:** optimization target mismatch is real (calibration vs decision-boundary quality).

**Hypotheses:**

- Distillation settings favored confidence shaping over minority boundary separation.
- KD/DKD objective weight/temperature schedule was not macro-F1-optimal.

---

## 4.5 NR-5: Teacher Stage-A validation did not predict hard-gate robustness

### 4.5.1 Outcome (evidence-backed)

Teachers strong on Stage-A val (~0.78–0.79 macro-F1) dropped substantially on:

- `classification_manifest_eval_only`
- `expw_full_manifest`

while performing better on `test_all_sources` than on ExpW/eval-only.

### 4.5.2 Interpretation

This is a core negative finding about evaluation transferability: in-distribution validation is insufficient as a deployment proxy.

### 4.5.3 Explanation status

**Evidence-backed:** evaluation distribution mismatch drives gap.

**Hypotheses:**

- Source filtering in Stage-A reduces exposure to hardest in-the-wild characteristics.
- Hard gates include shift/noise regimes not represented in Stage-A validation.

---

## 4.6 NR-6: Hard-gate failures concentrate in Fear/Disgust and persist under calibration

### 4.6.1 Outcome (evidence-backed)

From the low-performance investigation:

- ExpW/eval-only class distribution and confusion profile consistently penalize Fear/Disgust.
- FER2013 uniform-7 is balanced, yet Fear remains weak; therefore imbalance alone is insufficient explanation.
- Temperature scaling improved ECE/NLL but did not change macro-F1 ranking.

### 4.6.2 Interpretation

Class-wise representational mismatch under domain shift, not only confidence calibration, is the dominant issue.

### 4.6.3 Explanation status

**Evidence-backed:** calibration correction does not resolve low-class-separation failures.

**Hypotheses:**

- Preprocessing/crop/texture statistics mismatch for low-resolution or in-the-wild conditions.
- Label ambiguity and annotation noise in subtle expressions.

---

## 4.7 NR-7: KD+LP short-budget addendum did not show clear ExpW macro-F1 gain

### 4.7.1 Outcome (evidence-backed)

In the Feb-2026 short-budget KD baseline vs KD+LP study:

- Eval-only macro-F1 changed slightly upward for KD+LP.
- ExpW macro-F1 did not improve (slight decline in that screening).
- Some calibration terms improved.

### 4.7.2 Interpretation

LP-loss is not validated as a domain-shift macro-F1 improver under current short-budget settings.

---

## 5. Cross-Cutting Causal Structure (Evidence vs Hypotheses)

## 5.1 Evidence-backed causal statements

1. **Objective mismatch exists:** offline static gates and webcam replay can diverge.
2. **Gate necessity is validated:** without eval-only gate, regressive candidates could be mistakenly promoted.
3. **Class-specific fragility dominates:** Fear/Disgust are persistent weak links across hard gates.
4. **Calibration is not enough:** lower ECE/NLL does not guarantee macro-F1 or replay gain.

## 5.2 Hypothesis map (to be tested, not asserted)

1. **Buffer quality hypothesis:** pseudo-label noise and transition-frame contamination drive adaptation harm.
2. **NegL policy hypothesis:** medium-confidence negative-target definition may be mis-specified.
3. **Margin dynamics hypothesis:** adaptation changes top-1/top-2 margins in ways that destabilize temporal post-processing.
4. **Training-budget hypothesis:** short-run screens reveal risk but under-sample stable improvement regimes.

---

## 6. Threats to Validity

1. **External validity**
   - Some conclusions rely on single-session replay or limited run counts.

2. **Protocol variance risk**
   - Historical runs can differ in preprocessing defaults, crop pipeline, or temperature policy.

3. **Selection bias risk in reporting**
   - Mitigated by retaining both favorable and unfavorable outcomes in artifact-backed summaries.

4. **Underpowered screening**
   - Several studies are intentionally short-budget for fast triage; this prioritizes risk detection over final optimization.

---

## 7. Decision Rules for Next Phase (Pre-Registered)

Adopt the following progression for domain-shift adaptation claims:

1. **Offline safety first**
   - No candidate can be promoted if eval-only macro-F1/minority-F1 violate non-regression thresholds.

2. **Replay requirement**
   - No “deployment improvement” claim without identical-session replay gain using same labels and fixed scorer settings.

3. **One-knob design**
   - Change one variable per experiment (NegL policy, threshold, LR, update scope, etc.) to preserve causal interpretability.

4. **Replication requirement**
   - A candidate must pass at least two distinct labeled sessions before any generalized claim.

---

## 8. Prioritized Next-Step Experiments

### 8.1 Priority A (causal isolation)

1. Positives-only self-learning vs NegL-only vs combined (same session, same scorer).
2. Medium-confidence policy ablation:
   - ignore,
   - current predicted-label negative,
   - complementary-class negative.

### 8.2 Priority B (robustness)

3. Multi-session replay panel (different lighting/subjects) with fixed evaluation policy.
4. Buffer audit report per run:
   - class histogram,
   - confidence histogram,
   - pseudo-label agreement against manual labels (analysis-only).

### 8.3 Priority C (NL/NegL safety tuning)

5. Controlled low-weight sweeps with activity constraints (`applied_frac` monitoring).
6. Stop criteria triggered by minority-F1 regression even when aggregate calibration improves.

---

## 9. Final Conclusion

The project’s negative results are coherent, reproducible, and scientifically meaningful. They do not indicate failed research; they establish a rigorous boundary between offline metric optimization and true deployment-facing improvement. The most important conclusion is methodological: **a model update is not successful unless it passes both broad-distribution safety gates and fixed-protocol webcam replay criteria**. This principle, supported by the 2026-02-21 adaptation counterexample, is a central contribution of the project and should be retained as a formal evaluation standard in the final thesis defense.

---

## Appendix A. Evidence Path List

### A.1 Final integrated write-up

- `research/final report/final report version 2.md`

### A.2 Core issue analyses

- `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`
- `research/issue__teacher_hard_gates__20260209.md`
- `research/issue__teacher_metrics_interpretation__20260209.md`

### A.3 NL/NegL studies

- `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`
- `outputs/students/_compare*.md`

### A.4 Domain-shift adaptation artifacts

- `research/domain shift improvement via Self-Learning + Negative Learning plan/domain shift improvement via Self-Learning + Negative Learning report/domain shift improvement via Self-Learning + Negative Learning report.md`
- `demo/outputs/20260126_205446/score_results.json`
- `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`
- `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/`
- `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/`

### A.5 Early adaptation gate evidence

- `outputs/evals/students/_baseline_CE20251223_eval_only_test/reliabilitymetrics.json`
- `outputs/evals/students/FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358/reliabilitymetrics.json`
- `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/reliabilitymetrics.json`
