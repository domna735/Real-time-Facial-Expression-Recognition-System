# Teacher Metrics vs “Real” Performance — Interpretation Note

Date: 2026-02-09

## Why this document exists

A concern came up that the reported teacher results (macro-F1 ≈ 0.78–0.79) feel “not true” because real/mixed-domain tests (eval-only / ExpW / test_all_sources / webcam) look much worse.

This note clarifies **what those teacher metrics actually measure**, and why it is normal for them to be much higher than deployment-facing or mixed-source gates.

## 1) What the teacher metrics in the Final Report are (and are not)

### 1.1 What they are

The teacher results shown in Final Report Section 4.2 are copied directly from the on-disk artifacts:

- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/reliabilitymetrics.json`

These metrics are computed on the **Stage-A teacher validation split**, whose size and source composition are recorded in each run’s `alignmentreport.json`.

Example evidence (RN18):

- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/alignmentreport.json`
  - `val_rows = 18165`
  - `source_counts_after_filter = {ferplus, affectnet_full_balanced, rafdb_basic}`

### 1.2 What they are not

They are **not**:

- Performance on `Training_data_cleaned/classification_manifest_eval_only.csv`
- Performance on `Training_data_cleaned/expw_full_manifest.csv`
- Performance on `Training_data_cleaned/test_all_sources.csv`
- Performance under webcam domain shift

So it is expected that “real” stress tests can be much lower.

## 2) Why the teacher validation score can be “high” but real tests look “bad”

### Reason A — Evaluation distribution mismatch

The teacher’s reported metrics are on a validation split drawn from a filtered training distribution.

Hard gates like `eval_only`, `expw_full`, and `test_all_sources` are intentionally more diverse and shift-heavy; they include sources and conditions that are not matched to teacher training.

### Reason B — Source filtering reduces noise and simplifies the task

Stage-A teacher training uses a source filter.

RN18 evidence (alignment report):

- `rows_total_before_filter = 466,284`
- `rows_total_after_filter = 225,629`
- Sources actually present after filter: `affectnet_full_balanced`, `ferplus`, `rafdb_basic`

Even though the include list contains `expw_hq`, ExpW rows do not appear in `source_counts_after_filter` for Stage-A teachers. This makes the evaluation closer to “cleaner / more controlled” sources than ExpW.

### Reason C — Real-world deployment adds additional constraints

Real-time deployment changes what matters:

- stability (flicker) is a first-class KPI
- lighting / motion blur / camera sensor properties cause domain shift
- face detector + crop quality can degrade the input distribution

Many papers (and also teacher training) do not optimize for this deployment objective.

## 3) What evidence we already have for the “realistic” gates

A consolidated, artifact-backed snapshot for student checkpoints across four hard manifests exists:

- `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

This table shows that on the mixed-source gates (eval-only, ExpW, test_all_sources), raw macro-F1 is around ~0.46–0.49 for the evaluated student checkpoints.

Interpretation:

- This does **not** contradict the teacher’s ~0.79 macro-F1 on its in-distribution validation split.
- It indicates the hard gates are dominated by domain shift + noise + minority-class fragility (Fear/Disgust).

## 4) Conclusion

The teacher numbers reported in Section 4.2 are **true for the recorded evaluation split** (Stage-A validation, filtered sources).

If the question is “why are real/mixed-domain results not good?”, the primary explanation is **distribution mismatch** plus **deployment constraints** and **dataset noise**, not that the teacher results were fabricated.

## 5) Next actions (if we want to quantify the gap precisely)

To quantify “teacher vs real gates” directly (optional follow-up):

1. Evaluate teacher checkpoints on the same hard manifests (`eval_only`, `expw_full`, `test_all_sources`) using the same evaluation script.
2. Put results into the same table format as `overall_summary__20260208`.
3. Compare per-class F1 (Fear/Disgust) to confirm whether failure modes are shared between teacher and student.
