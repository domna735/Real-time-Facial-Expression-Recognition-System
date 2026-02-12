# Low-Performance Dataset Investigation Report
Generated: 2026-02-08 20:36:00

## Scope
This report investigates unexpectedly weak performance on the following evaluation datasets:
- `classification_manifest_eval_only`
- `expw_full_manifest`
- `test_fer2013_uniform_7`

## Evidence (suite artifacts)
- Suite time: `2026-02-08 19:26:04`
- Suite dir: `outputs/benchmarks/offline_suite__20260208_192604`
- Index: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`
- Full results CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Per-class F1 CSV (generated): `outputs/benchmarks/offline_suite__20260208_192604/bad_datasets__per_class_f1__20260208_203600.csv`
- Dataset audit CSV (generated): `outputs/benchmarks/offline_suite__20260208_192604/bad_datasets__audit__20260208_203600.csv`

## High-level observation
Across these datasets, macro-F1 is pulled down mainly by consistent weak classes (commonly `Disgust` and `Fear`).
Temperature scaling improves calibration (ECE / NLL) but does not change macro-F1, suggesting the main issue is not confidence calibration but domain shift / label noise / class ambiguity.

## Key findings (evidence-based)
1) **ExpW and eval-only are long-tail on `Fear` / `Disgust` (macro-F1 is very sensitive).**
	- `expw_full_manifest` test label counts show `Fear:109/9179 (~1.2%)`, `Disgust:399/9179 (~4.3%)`.
	- `classification_manifest_eval_only` test label counts show `Fear:351/11890 (~3.0%)`, `Disgust:741/11890 (~6.2%)`.
	- Across *all* evaluated models, these classes are consistently weak:
	  - `expw_full_manifest`: `Disgust` min/max F1 `0.098–0.176`, `Fear` min/max F1 `0.113–0.215`.
	  - `classification_manifest_eval_only`: `Disgust` min/max F1 `0.219–0.283`.

2) **FER2013 is balanced but `Fear` remains consistently weak → not a class-imbalance problem.**
	- `test_fer2013_uniform_7` has `2000` samples per class (perfectly balanced).
	- Yet `Fear` min/max F1 is only `0.156–0.193` across all models.
	- This points to a systematic mismatch (domain / preprocessing / label semantics), not just scarcity.

3) **Data integrity checks look OK (not a broken manifest/path issue).**
	- Sampled file existence checks are `2000/2000` OK for all three datasets.
	- ExpW has bboxes present for `100%` of test rows, so the issue is not “missing bbox”.
	- FER2013 has `0%` bbox fields present (expected given dataset format), which increases the chance of a preprocessing mismatch versus aligned/cropped face datasets.

4) **Calibration is not the primary driver of low macro-F1.**
	- Temperature scaling reduces ECE and NLL substantially (see summary table), but macro-F1 does not change.
	- That suggests the main failure mode is *ranking / decision boundary / representation* under domain shift, not probability calibration.

## Dataset audits (manifest-level)
### classification_manifest_eval_only
- Manifest: `C:/Real-time-Facial-Expression-Recognition-System_v2_restart/Training_data_cleaned/classification_manifest_eval_only.csv`
- Eval split: `test`
- Rows (split/chosen): `11890`
- BBox present: `10116` (85.08%)
- File existence check (sampled): `2000/2000` (100.00%)
- Top labels: Neutral:3619, Happy:3535, Sad:1603, Surprise:1152, Angry:889, Disgust:741, Fear:351
- Top sources: expw_full:6780, expw_hq:3336, rafml_argmax:982, rafdb_compound_mapped:792

### expw_full_manifest
- Manifest: `C:/Real-time-Facial-Expression-Recognition-System_v2_restart/Training_data_cleaned/expw_full_manifest.csv`
- Eval split: `test`
- Rows (split/chosen): `9179`
- BBox present: `9179` (100.00%)
- File existence check (sampled): `2000/2000` (100.00%)
- Top labels: Neutral:3489, Happy:3053, Sad:1056, Surprise:706, Disgust:399, Angry:367, Fear:109
- Top sources: expw_full:9179

### test_fer2013_uniform_7
- Manifest: `C:/Real-time-Facial-Expression-Recognition-System_v2_restart/Training_data_cleaned/test_fer2013_uniform_7.csv`
- Eval split: `test`
- Rows (split/chosen): `14000`
- BBox present: `0` (0.00%)
- File existence check (sampled): `2000/2000` (100.00%)
- Top labels: Angry:2000, Disgust:2000, Fear:2000, Happy:2000, Neutral:2000, Sad:2000, Surprise:2000
- Top sources: fer2013_uniform_7:14000

## Model results summary (these datasets only)
model_kind | model | dataset | raw_macro_f1 | raw_acc | raw_ece | raw_nll | ts_temp | ts_ece | ts_nll | out_dir
---|---|---:|---:|---:|---:|---:|---:|---:|---:|---
student | mobilenetv3_large_100_img224_seed1337_CE_20251223_225031 | classification_manifest_eval_only | 0.485878 | 0.567368 | 0.183799 | 1.647025 | 2.606 | 0.059181 | 1.228754 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/classification_manifest_eval_only`
student | mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722 | classification_manifest_eval_only | 0.437137 | 0.510008 | 0.375097 | 2.466666 | 3.518 | 0.048101 | 1.352879 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/classification_manifest_eval_only`
student | mobilenetv3_large_100_img224_seed1337_KD_20251231_074714 | classification_manifest_eval_only | 0.438958 | 0.519849 | 0.338308 | 2.086811 | 3.131 | 0.032386 | 1.283128 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_KD_20251231_074714/classification_manifest_eval_only`
teacher | B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224 | classification_manifest_eval_only | 0.392565 | 0.470732 | 0.505091 | 6.821328 | 5.000 | 0.267513 | 1.694643 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/classification_manifest_eval_only`
teacher | CNXT_convnext_tiny_seed1337_stageA_img224 | classification_manifest_eval_only | 0.388980 | 0.440875 | 0.537715 | 7.713847 | 5.000 | 0.316119 | 1.844298 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/CNXT_convnext_tiny_seed1337_stageA_img224/classification_manifest_eval_only`
teacher | RN18_resnet18_seed1337_stageA_img224 | classification_manifest_eval_only | 0.372670 | 0.427250 | 0.541873 | 10.481448 | 5.000 | 0.390783 | 2.318526 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/RN18_resnet18_seed1337_stageA_img224/classification_manifest_eval_only`
student | mobilenetv3_large_100_img224_seed1337_CE_20251223_225031 | expw_full_manifest | 0.482120 | 0.657697 | 0.189538 | 1.425482 | 2.533 | 0.025780 | 0.993338 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/expw_full_manifest`
student | mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722 | expw_full_manifest | 0.459529 | 0.622944 | 0.254957 | 1.820043 | 2.984 | 0.036783 | 1.102717 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/expw_full_manifest`
student | mobilenetv3_large_100_img224_seed1337_KD_20251231_074714 | expw_full_manifest | 0.469368 | 0.640484 | 0.215957 | 1.560278 | 2.715 | 0.021513 | 1.034233 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_KD_20251231_074714/expw_full_manifest`
teacher | B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224 | expw_full_manifest | 0.406651 | 0.584486 | 0.398763 | 5.674561 | 5.000 | 0.203973 | 1.401287 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/expw_full_manifest`
teacher | CNXT_convnext_tiny_seed1337_stageA_img224 | expw_full_manifest | 0.382112 | 0.511167 | 0.468350 | 6.667708 | 5.000 | 0.260479 | 1.619482 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/CNXT_convnext_tiny_seed1337_stageA_img224/expw_full_manifest`
teacher | RN18_resnet18_seed1337_stageA_img224 | expw_full_manifest | 0.374009 | 0.498747 | 0.465685 | 8.207039 | 5.000 | 0.304591 | 1.883318 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/RN18_resnet18_seed1337_stageA_img224/expw_full_manifest`
student | mobilenetv3_large_100_img224_seed1337_CE_20251223_225031 | test_fer2013_uniform_7 | 0.497356 | 0.524143 | 0.386376 | 3.824325 | 5.000 | 0.041435 | 1.381818 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/test_fer2013_uniform_7`
student | mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722 | test_fer2013_uniform_7 | 0.502030 | 0.528857 | 0.398485 | 3.630525 | 5.000 | 0.035276 | 1.377141 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/test_fer2013_uniform_7`
student | mobilenetv3_large_100_img224_seed1337_KD_20251231_074714 | test_fer2013_uniform_7 | 0.499439 | 0.523286 | 0.397002 | 4.482403 | 5.000 | 0.091105 | 1.423763 | `outputs/benchmarks/offline_suite__20260208_192604/evals/student/mobilenetv3_large_100_img224_seed1337_KD_20251231_074714/test_fer2013_uniform_7`
teacher | B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224 | test_fer2013_uniform_7 | 0.478930 | 0.510857 | 0.477329 | 7.348342 | 5.000 | 0.289983 | 1.707044 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/test_fer2013_uniform_7`
teacher | CNXT_convnext_tiny_seed1337_stageA_img224 | test_fer2013_uniform_7 | 0.488917 | 0.520714 | 0.467989 | 7.303158 | 5.000 | 0.280367 | 1.696540 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/CNXT_convnext_tiny_seed1337_stageA_img224/test_fer2013_uniform_7`
teacher | RN18_resnet18_seed1337_stageA_img224 | test_fer2013_uniform_7 | 0.485123 | 0.513857 | 0.468480 | 10.370565 | 5.000 | 0.368037 | 2.213466 | `outputs/benchmarks/offline_suite__20260208_192604/evals/teacher/RN18_resnet18_seed1337_stageA_img224/test_fer2013_uniform_7`

## Per-class F1 highlights (where macro-F1 is lost)
For each dataset, the table below lists the minimum and maximum per-class F1 observed across all evaluated models (teachers + students).

### classification_manifest_eval_only (raw)
class | min_f1 | max_f1 | note
---|---:|---:|---
Angry | 0.332 | 0.455 | 
Disgust | 0.219 | 0.283 | consistently weak across all models
Fear | 0.232 | 0.408 | 
Happy | 0.515 | 0.678 | 
Neutral | 0.470 | 0.602 | 
Sad | 0.398 | 0.477 | 
Surprise | 0.383 | 0.498 | 

### expw_full_manifest (raw)
class | min_f1 | max_f1 | note
---|---:|---:|---
Angry | 0.322 | 0.487 | 
Disgust | 0.098 | 0.176 | consistently weak across all models
Fear | 0.113 | 0.215 | consistently weak across all models
Happy | 0.719 | 0.794 | 
Neutral | 0.518 | 0.701 | 
Sad | 0.399 | 0.489 | 
Surprise | 0.393 | 0.546 | 

### test_fer2013_uniform_7 (raw)
class | min_f1 | max_f1 | note
---|---:|---:|---
Angry | 0.463 | 0.487 | 
Disgust | 0.392 | 0.474 | 
Fear | 0.156 | 0.193 | consistently weak across all models
Happy | 0.726 | 0.789 | 
Neutral | 0.477 | 0.498 | 
Sad | 0.358 | 0.390 | 
Surprise | 0.724 | 0.746 | 

## Likely causes (hypotheses ranked by evidence)
1) **Domain shift / label ambiguity in real-world sets (ExpW + mixed-source eval-only):** low F1 concentrates in `Disgust`/`Fear`, which are subtle and frequently confused in-the-wild.
2) **Class imbalance / long-tail:** if the manifest audit shows very small counts for `Disgust`/`Fear`, macro-F1 will drop quickly even if accuracy stays moderate.
3) **Annotation noise / weak labels:** ExpW is known to be noisy; mixed-source eval-only contains diverse acquisition/quality.
4) **Preprocessing mismatch:** FER2013 images differ strongly (low-res, grayscale, different cropping/alignment). Our pipeline forces resizing to 224 and applies CLAHE; this may help/hurt depending on domain.
5) **Model capacity + training objective effects:** KD/DKD may improve certain domains but degrade others if teacher is biased; CE appears more stable across domains in this suite.

## Next tests to pinpoint the root cause (actionable)
- **Per-source breakdown for eval-only:** run the same model evaluation but grouped by `source` to see which sources dominate the drop.
- **Ablate CLAHE on FER2013/ExpW:** re-evaluate with `use_clahe=false` to confirm whether CLAHE is helping or harming these domains.
- **Error analysis on weak classes:** sample misclassified `Disgust`/`Fear` images and inspect whether they are ambiguous/occluded/low-quality or mislabeled.
- **Re-weight / focal loss / class-balanced sampling:** specifically target `Disgust` and `Fear` during training or finetuning.
- **Domain-adaptive augmentation:** add grayscale/noise/low-res augmentation to better match FER2013-like conditions.
