# Paper vs Us (protocol-matched where possible)

Date: 2026-02-08

Goal: provide a professor-facing, *fair* comparison table. “Fair” means we only claim comparability when the dataset split, label-space, and evaluation protocol match.

Primary evidence sources (on disk):

- Our offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Our suite “best by dataset” (raw acc): `outputs/benchmarks/offline_suite__20260208_192604/benchmark_best_by_dataset__raw_acc.csv`
- Paper text extracts: `outputs/paper_extract/*.txt`

Important interpretation (to avoid misleading “apples-to-oranges” comparisons):

- Some of our strongest-looking numbers (e.g., teacher macro-F1 ≈ 0.78–0.79) are **Stage-A in-distribution validation** metrics after source filtering.
- Some of our lowest-looking numbers (e.g., `eval_only`, `expw_full`) are **hard/mixed-domain gates** designed to stress domain shift and label noise.
- These are both valid results, but they are **not the same evaluation distribution**, so large gaps are expected and are not evidence of “forgery”.

Hard-gate teacher benchmark (same teacher checkpoints on the hard gates):

- Summary table: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
- Write-up note: `research/issue__teacher_hard_gates__20260209.md`

## Comparison method (scientific + fair)

Rules used in this table:

1) **Comparable = Yes** only if the paper and our run match on:
   - dataset + split definition (official split vs custom split)
   - label space (7-class mapping and label remapping)
   - evaluation protocol (single-crop vs ten-crop/TTA; face crop/alignment; preprocessing)
   - metric definition (accuracy vs macro-F1; weighted vs unweighted)

2) If any of the above differs, mark **Partial** (same dataset family but protocol mismatch) or **No** (different dataset/metric).

3) We separate results into evaluation regimes so numbers cannot be misread:
   - **In-distribution** (Stage-A validation after filtering)
   - **Mixed-source benchmark** (stress test but closer to training mixture)
   - **Cross-dataset / in-the-wild hard gates** (domain shift)

## Our results (by evaluation regime)

This section exists to prevent mixing “in-distribution validation” with “hard-gate domain shift” numbers.

Teacher (Stage A, img224):

- In-distribution Stage-A validation (`val_rows = 18165`):
  - `outputs/teachers/*/reliabilitymetrics.json` (macro-F1 ≈ 0.78–0.79)
- Hard gates (same checkpoints on `eval_only` / `expw_full` / `test_all_sources`):
  - Summary: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`

Student (MobileNetV3-Large; CE vs KD vs DKD) hard gates:

- Summary: `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

Interpretation:

- Large gaps between Stage-A val and hard gates are **expected** under domain shift and label noise.
- When comparing to papers, always compare using the closest matching regime (prefer official splits when available).

## One-page comparison table

| Dataset | Paper metric (as stated) | Paper protocol notes (quotable) | Our closest matching artifact | Our metric | Comparable? |
| --- | ---: | --- | --- | ---: | --- |
| FER2013 (official split; PublicTest/PrivateTest from `fer2013.csv`) | Accuracy: **73.28%** (single network) | “adhere to the official training, validation, and test sets” + “tested using standard ten-crop averaging” + no extra training data | Summary table: `outputs/benchmarks/fer2013_official_summary__20260209/fer2013_official_summary.md` (backed by per-run `reliabilitymetrics.json`) | Best student (DKD_20251229_223722): PublicTest acc **0.613820**, macro-F1 **0.553508**; PrivateTest acc **0.608247**, macro-F1 **0.539047** | **Partial** — official split matches, but protocol mismatch vs paper (paper: ten-crop / TTA; ours: single-crop) |
| FER2013 (folder split: `Training_data/FER2013/{train,test}`) | N/A (dataset packaging differs from ICML official split) | Folder-based split (not ten-crop) | `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json` | DKD_20251229_223722: acc **0.604904**, macro-F1 **0.540220** | **Partial** — same label space, but not the ICML “official train/val/test” protocol; use as an additional controlled benchmark |
| RAF-DB | Table 5 “Padding” whole-face accuracy: **82.69%** | Paper reports RAF-DB test accuracy with/without padding; “with padding” improves performance | Suite dataset: `test_rafdb_basic` | Best student (CE_20251223_225031): acc **0.862777**, macro-F1 **0.791656** | **Partial** — likely similar label-space, but preprocessing differs (paper uses explicit padding; our pipeline may include CLAHE + cleaned manifests) |
| FER+ (FERPlus) | Original dataset accuracy: **0.752**; masked-augmented accuracy: **0.747** | Extracted from paper abstract: original 0.752 vs masked augmented 0.747 | Suite dataset: `test_ferplus` | Best student (CE_20251223_225031): acc **0.842224**, macro-F1 **0.738838** | **No** — paper’s dataset variants/protocol differ; also FERPlus label handling can differ across works |
| ExpW | Table 6 whole-face accuracy: **71.90%** (ExpW testing set) | Paper notes ExpW has worse generalization, likely due to labeling quality | Suite dataset: `expw_full_manifest` | Best student (CE_20251223_225031): acc **0.657697**, macro-F1 **0.482120** | **Partial** — same dataset name family, but label mapping/cropping rules may differ; treat as domain/generalization test |
| AffectNet | Table 7 (Weighted-Loss) derived macro-F1: **0.625** (Top-1, skew-normalized) | Paper reports per-class F1; our derived macro-F1 is computed in `outputs/paper_extract/affectnet__table7_weightedloss_macro_f1.md` | Suite dataset: `test_affectnet_full_balanced` | Best student (CE_20251223_225031): acc **0.822439**, macro-F1 **0.822597** | **No** — different subset (balanced) and different metric definition/reporting |

## Quotes / evidence pointers

FER2013 paper (extract: `outputs/paper_extract/Facial Emotion Recognition State of the Art Performance on FER2013.txt`):

- “state-of-the-art single-network accuracy of 73.28 % on FER2013”
- “adhere to the official training, validation, and test sets”
- “tested using standard ten-crop averaging”

RAF/FER+/ExpW face-regions paper (extract: `outputs/paper_extract/Expression Analysis Based on Face Regions in Read-world Conditions.txt`):

- Table 5 RAF-DB padding vs non-padding: whole face (padding) **82.69**
- Table 6 whole face accuracies: FER+ **81.93**, RAF-DB **82.69**, ExpW **71.90** (table text extraction is slightly misaligned; verify in PDF if needed)

FERPlus masked-augmentation paper (extract: `outputs/paper_extract/Facial_Emotion_Recognition_Using_Masked-Augmented_FERPlus.txt`):

- Abstract includes: “accuracy on the original dataset, which was 0.752 … masked augmented dataset … accuracy of 0.747.”

AffectNet paper macro-F1 derivation note:

- `outputs/paper_extract/affectnet__table7_weightedloss_macro_f1.md`

## How to run strict FER2013 (PublicTest / PrivateTest) evaluation

Prereq: you must already have a local Kaggle/ICML `fer2013.csv` (license-restricted).

Status (this workspace, updated 2026-02-09):

- Source CSV present: `challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013/fer2013.csv`
- Converted official manifests + images:
  - `Training_data/FER2013_official_from_csv/manifest__publictest.csv` (n=3589)
  - `Training_data/FER2013_official_from_csv/manifest__privatetest.csv` (n=3589)
- Student eval artifacts (one per model × split) under `outputs/evals/students/fer2013_official__*__*test__20260209/`.
- Consolidated summary:
  - `outputs/benchmarks/fer2013_official_summary__20260209/fer2013_official_summary.md`

1) Convert `fer2013.csv` → images + manifest (PublicTest):

- `python tools/data/convert_fer2013_csv_to_manifest.py --fer2013-csv <PATH_TO_fer2013.csv> --usage PublicTest`

1) Evaluate our best student checkpoint on that manifest:

- Best DKD checkpoint (from existing runs): `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/best.pt`
- Command:
  - `python scripts/eval_student_checkpoint.py --checkpoint outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/best.pt --eval-manifest Training_data/FER2013_official_from_csv/manifest__publictest.csv --eval-split test --eval-data-root .`

This writes `reliabilitymetrics.json` containing **accuracy** and **macro-F1** under `outputs/evals/students/`.

If you want PrivateTest too:

- Re-run the converter with `--usage PrivateTest`, then evaluate on `manifest__privatetest.csv`.

If you also want to approximate the paper’s “ten-crop” test-time evaluation, we should add a dedicated evaluation mode (ten-crop) and report it separately, because it is not equivalent to single-crop evaluation.
