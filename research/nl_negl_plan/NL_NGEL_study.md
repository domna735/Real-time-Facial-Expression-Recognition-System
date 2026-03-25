# NL / NegL Study Notes (Student Training)

This file records *what was actually run* and *what we learned* while validating NegL/NL ideas on the student training pipeline.

## Implementation note (NL v2: stable default)

NL now supports two variants (selected via `--nl-kind`):
- `proto` (default): **prototype memory (32–64 dim)** + **momentum smoothing** + **consistency-gated** auxiliary loss. This can be used **without** NegL.
- `negl_gate` (legacy): learned per-sample gate that scales NegL sample weights (this is what the earlier `KD+NegL+NL` run used).

NL(proto) representation source:
- `--nl-embed penultimate` (default): use student penultimate features (recommended; makes gating non-degenerate)
- `--nl-embed logits` (legacy): use student logits as embedding source (stable but tended to become nearly inactive)

Runner example (NL proto, no NegL):
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -UseNL -NLKind proto -NLEmbed penultimate -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.2 -NLWeight 0.1 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp`

Runner example (legacy NL gate for NegL):
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -UseNegL -UseNL -NLKind negl_gate -NLHiddenDim 32 -NLLayers 1 ...`

## Experiment: KD 5 epochs + NegL (entropy-gated)

Date/run stamp:
- 2025-12-28 (`*_20251228_233720`)

Purpose:
- Validate NegL wiring + logging in a short, controlled KD ablation.
- Check early signals for calibration (ECE/NLL) and minority-class behavior.

### Setup

Student:
- Model: `mobilenetv3_large_100`
- Image size: 224
- Seed: 1337

Data:
- Train manifest: `Training_data_cleaned/classification_manifest_hq_train.csv`
- Data root: `Training_data_cleaned`

Teacher supervision:
- Softlabels dir:
	- `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/`
- KD hyperparams: temperature=2, alpha=0.5

Runtime stability:
- Batch size: 128
- Workers: 2
- CLAHE: on
- AMP: on

NegL config:
- Enabled: yes
- Weight: 0.05
- Ratio: 0.5
- Gate: entropy
- Entropy threshold: 0.7

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7 -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 2 -UseClahe -UseAmp`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720/`
	- Key files: `history.json`, `reliabilitymetrics.json`, `calibration.json`, `best.pt`

### Results (validation, final epoch)

From `reliabilitymetrics.json` (epoch 4 / end of KD run):

| Metric | Value |
|---|---:|
| Accuracy | 0.7224 |
| Macro-F1 | 0.7198 |
| Raw NLL | 1.7801 |
| Raw ECE | 0.2106 |
| TS global temperature | 4.3918 |
| TS NLL | 0.8085 |
| TS ECE | 0.0398 |

Per-class F1 (final epoch):
- Angry 0.7130
- Disgust 0.6317
- Fear 0.7568
- Happy 0.7550
- Sad 0.7036
- Surprise 0.7646
- Neutral 0.7139

NegL application stats (from `history.json`):
- Entropy gate is very selective at threshold=0.7.
- NegL applied fraction over epochs:
	- epoch 0: 0.03196
	- epoch 1: 0.00970
	- epoch 2: 0.00720
	- epoch 3: 0.00654
	- epoch 4: 0.00583

### Interpretation

What this run proves:
- NegL integration is functioning end-to-end (loss computed, gated, and logged).
- The training loop remains stable on Windows under the chosen loader settings.

What this run does *not* yet prove:
- Whether NegL improves anything vs baseline, because we still need a **baseline KD 5-epoch run (NegL off)** under identical conditions. (Now completed; see below.)

Important observation:
- With `negl_entropy_thresh=0.7`, the model’s average entropy is far below the threshold (mean entropy falls from ~0.28 to ~0.08), so NegL affects <~3.2% of samples early and <~0.6% late.
- This likely makes the NegL signal too weak to move metrics noticeably; the run is a good “wiring validation” but a weak stress test.

### Notes on CE/DKD stages in this run

The runner executed CE/KD/DKD blocks, but the chosen epoch counts mean:
- CE: `-CeEpochs 0` → CE outputs may be empty (no training steps).
- DKD: `-DkdEpochs 0` means “0 additional epochs”. DKD resumed at `start_epoch=5` with total epochs=5, so it performed no training (DKD outputs empty).

### Next actions (to make this scientifically comparable)

1) Run baseline KD 5 epochs (NegL off), identical settings:
- `scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 2 -UseClahe -UseAmp`

2) Compare baseline KD 5ep vs KD+NegL 5ep:
- `python tools/diagnostics/compare_student_runs.py <baseline_kd_dir> outputs/students/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720 --out outputs/students/_compare_kd5_vs_negl_kd5.md`

3) If we want NegL to actually “bite”, run a small gate sweep (keep everything else fixed):
- Lower entropy threshold (e.g., 0.3–0.5) and re-check `applied_frac`.
- Optional control: `--negl-gate none` (pure ratio-only) for 1 short run, to sanity-check sign/magnitude.

---

## Experiment: KD 5 epochs baseline (NegL off)

Date/run stamp:
- 2025-12-29 (`*_20251229_182119`)

Purpose:
- Create the fair KD-only baseline to compare against KD+NegL (same epoch budget and settings).

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

Comparison table:
- `outputs/students/_compare_kd5_vs_negl5.md`

Quick outcome (from comparison table):
- KD-only (5ep): acc 0.7284, macro-F1 0.7266, TS ECE 0.0271
- KD+NegL (5ep): acc 0.7224, macro-F1 0.7198, TS ECE 0.0398

---

## Experiment: KD 5 epochs + NL(proto) (NegL off)

Date/run stamp:
- 2025-12-30 (`*_20251230_004048`) (completed)

Purpose:
- First stability check for the new NL(proto) mechanism without mixing in NegL.
- Keep the epoch budget identical to the KD-only baseline (5 epochs) for fair comparison.

NL(proto) config:
- dim=32
- momentum=0.9
- consistency_thresh=0.2
- weight=0.1

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.2 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_vs_kd5.md`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251230_004048/`
- Logs:
	- `outputs/students/Log and test/_logs_20251230_004048/`
- Compare table:
	- `outputs/students/_compare_kd5_nlproto_vs_kd5.md`

### Results (validation, final epoch)

From compare table:
- KD-only baseline (5ep): acc 0.728363, macro-F1 0.726648, TS ECE 0.027051, TS NLL 0.783856, minority-F1 0.697342
- KD+NL(proto) (5ep): acc 0.729573, macro-F1 0.728076, TS ECE 0.042676, TS NLL 0.796150, minority-F1 0.694379

NL(proto) stats (from `history.json`, final epoch):
- train_nl_loss: 1.68e-05
- applied_frac: 7.16e-05
- sim_mean: 0.9754

### Interpretation

What this suggests:
- The NL(proto) run is **stable** (no collapse like the legacy NegL-gated NL run).
- However, with `consistency_thresh=0.2`, NL is effectively **inactive** (`applied_frac ~ 0.007%`), so we should not expect it to move metrics much yet.

Next action to make NL(proto) “bite” while staying stable:
- Lower the consistency threshold (e.g., 0.05–0.10) and re-run the same 5-epoch KD ablation.
- Keep NegL off until we see that NL(proto) actually applies to a meaningful fraction of samples and does not destabilize KD.

### Follow-up: KD 5 epochs + NL(proto) consistency threshold sweep

Goal:
- Increase NL(proto) `applied_frac` (make it active) while preserving stability.

Baseline reference (same as before):
- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

#### Run A — thr=0.10

Date/run stamp:
- 2025-12-31 (`*_20251231_031407`)

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.10 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p10_vs_kd5.md`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_031407/`
- Compare table:
	- `outputs/students/_compare_kd5_nlproto_thr0p10_vs_kd5.md`

Results (final epoch):
- From compare table:
	- KD+NL(proto, thr=0.10): acc 0.731944, macro-F1 0.730627, TS ECE 0.032719, TS NLL 0.762842, minority-F1 0.698116
- NL(proto) stats (from `history.json` → `nl`):
	- train_nl_loss 1.70e-05
	- applied_frac 1.38e-04 (~0.0138%)
	- sim_mean 0.9883

#### Run B — thr=0.05

Date/run stamp:
- 2025-12-31 (`*_20251231_071347`)

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.05 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p05_vs_kd5.md`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_071347/`
- Compare table:
	- `outputs/students/_compare_kd5_nlproto_thr0p05_vs_kd5.md`

Results (final epoch):
- From compare table:
	- KD+NL(proto, thr=0.05): acc 0.727061, macro-F1 0.724867, TS ECE 0.022909, TS NLL 0.762270, minority-F1 0.691834
- NL(proto) stats (from `history.json` → `nl`):
	- train_nl_loss 1.61e-05
	- applied_frac 2.29e-04 (~0.0229%)
	- sim_mean 0.9948

#### Run C — thr=0.005

Date/run stamp:
- 2025-12-31 (`*_20251231_074714`)

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.005 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p005_vs_kd5.md`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_074714/`
- Compare table:
	- `outputs/students/_compare_kd5_nlproto_thr0p005_vs_kd5.md`

Results (final epoch):
- From compare table:
	- KD+NL(proto, thr=0.005): acc 0.732223, macro-F1 0.731533, TS ECE 0.030852, TS NLL 0.763651, minority-F1 0.702075
- NL(proto) stats (from `history.json` → `nl`):
	- train_nl_loss 1.98e-06
	- applied_frac 2.77e-04 (~0.0277%)
	- sim_mean 0.9995

### Interpretation (updated)

- The runs remain **stable**, but NL(proto) is still effectively **inactive** (applied_frac stays <0.03%) even after reducing the threshold by 40×.
- The cosine similarity becomes extremely high (sim_mean ~0.99–0.9995), so the current gating condition almost never triggers.
- Likely cause: the current NL(proto) embedding is derived from student logits, which makes prototype alignment “too easy” and yields a near-zero auxiliary signal.

Next action (before moving to NegL):
- Change NL(proto) to use a richer embedding source (e.g., student penultimate features) instead of logits, then re-run the same KD 5-epoch test and re-check `applied_frac`.

---

## Experiment: KD 5 epochs + NL(proto, penultimate embedding) (NegL off)

Date/run stamp:
- 2025-12-31 (`*_20251231_155841`) (completed)

Purpose:
- Validate that switching NL(proto) embedding from logits -> penultimate features makes the gate meaningfully active (non-trivial `applied_frac`).

NL(proto) config:
- embed=penultimate
- dim=32
- momentum=0.9
- consistency_thresh=0.2
- weight=0.1

Command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 64 -NumWorkers 8 -UseNL -NLKind proto -NLEmbed penultimate -NLConsistencyThresh 0.2 -NLWeight 0.1`

Artifacts:
- KD run folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_155841/`
- Logs:
	- `outputs/students/Log and test/_logs_20251231_155841/`
- Compare table:
	- `outputs/students/_compare_kd5_nlproto_penultimate_thr0p2_vs_kd5.md`

### Results (validation, final epoch)

From compare table:
- KD-only baseline (5ep): acc 0.728363, macro-F1 0.726648, TS ECE 0.027051, TS NLL 0.783856, minority-F1 0.697342
- KD+NL(proto, penultimate, thr=0.2): acc 0.726689, macro-F1 0.724393, TS ECE 0.039511, TS NLL 0.799723, minority-F1 0.691421

NL(proto) stats (from `history.json` → `nl`):
- applied_frac by epoch: [0.041732, 0.000138, 0.000033, 0.000010, 0.000000]
- sim_mean (final epoch): 0.9860

### Interpretation

- Penultimate embedding makes NL(proto) clearly active early in training (epoch 0 applied_frac ~4.17%), confirming the original “logits are too degenerate” hypothesis.
- However, the auxiliary signal rapidly vanishes after the first epoch (applied_frac -> ~0 by epoch 4), likely because prototypes align quickly and `(1 - sim)` falls below the threshold for almost all samples.

Next action (still within NL-only, before NegL):
- Tune NL(proto) to keep a steady applied fraction across epochs (e.g., lower `consistency_thresh`, reduce momentum, or adjust loss weighting), then re-run the same 5-epoch KD check.

---

## Experiment: KD 5 epochs + NegL + NL (learned NegL gate)

Date/run stamp:
- 2025-12-29 (`*_20251229_194408`) (completed)

Note:
- The earlier run `*_20251229_191103` was started before the runner correctly passed `--use-nl`, so it is **KD+NegL only** (no NL). Use `*_20251229_194408` for the real KD+NegL+NL result.

What “NL” means here (current implementation):
- A small learned gate (MLP) produces a per-sample weight in [0,1] to scale the NegL term.
- Input features: normalized entropy, max-prob, margin (top1-top2), step fraction.

Setup:
- Same student/data/softlabels as the other KD 5-epoch runs.
- NegL config kept identical to the 2025-12-28 run (w=0.05, ratio=0.5, gate=entropy, thresh=0.7).
- NL config:
	- hidden_dim=32, layers=1

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7 -UseNL -NLHiddenDim 32 -NLLayers 1`

Artifacts:
- KD run folder (expected):
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_194408/`
- Logs:
	- `outputs/students/Log and test/_logs_20251229_194408/`

Auto-compare (planned):
- Compare against KD-only baseline:
	- baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`
	- out: `outputs/students/_compare_kd5_negl_nl_vs_kd5.md`

### Results (validation, final epoch)

From `reliabilitymetrics.json` (epoch 4 / end of KD run):

| Metric | Value |
|---|---:|
| Accuracy | 0.5402 |
| Macro-F1 | 0.5204 |
| Raw NLL | 1.5776 |
| Raw ECE | 0.2349 |
| TS global temperature | 2.4015 |
| TS NLL | 1.2021 |
| TS ECE | 0.0315 |

Per-class F1 (final epoch):
- Angry 0.1961
- Disgust 0.6371
- Fear 0.6985
- Happy 0.6399
- Sad 0.4970
- Surprise 0.6178
- Neutral 0.3563

NegL + NL logging (from `history.json`):
- NegL applied fraction increased over epochs (entropy mean stayed high):
	- epoch 0: 0.03267 (entropy mean 0.303)
	- epoch 4: 0.04159 (entropy mean 0.342)
- NL gate stats:
	- epoch 0: gate_mean 0.525, gate_applied_mean 0.471
	- epoch 4: gate_mean 0.527, gate_applied_mean 0.519

### Comparison vs KD-only baseline

Auto-compare output:
- `outputs/students/_compare_kd5_negl_nl_vs_kd5.md`

Summary (from comparison table):
- KD-only (5ep): acc 0.7284, macro-F1 0.7266, TS ECE 0.0271, minority-F1 0.6973
- KD+NegL+NL (5ep): acc 0.5402, macro-F1 0.5204, TS ECE 0.0315, minority-F1 0.3498

### Interpretation

What this run proves:
- The NL gate is wired end-to-end (flags passed, gate is logged, and the compare tool detects NL=on).

What this run strongly suggests:
- The current NL gating setup is **not safe** under these settings: it causes a very large regression in accuracy/Macro-F1 and minority-F1.
- The model remains much more uncertain than the KD-only/NegL-only runs (entropy_mean ~0.34 here vs ~0.08 by epoch 4 in the NegL-only run), which likely increases the NegL pressure and can amplify instability.

Next actions (recommended order):
1) Do **not** proceed to DKD+NegL+NL until NL is stabilized.
2) Proceed to **DKD+NegL (without NL)** next (keeps the ablation plan moving while isolating NL debugging).
3) When returning to NL debugging, first try a safer variant:
	- Keep `--use-negl` on but turn entropy gate off (`--negl-gate none`) for a short run to see if the interaction is coming from entropy gating.
	- Or reduce NegL pressure (e.g., `--negl-weight 0.01` and/or `--negl-ratio 0.1`) and re-test NL.

---

## Experiment: DKD +5 epochs (resume from KD baseline; NegL off)

Date/run stamp:
- 2025-12-29 (`*_20251229_223722`) (completed)

Purpose:
- Establish a DKD baseline that is directly comparable to DKD+NegL by resuming from the same KD checkpoint.

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 0 -DkdEpochs 5 -DkdResumeFrom outputs\students\KD\mobilenetv3_large_100_img224_seed1337_KD_20251229_182119\checkpoint_last.pt -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp`

Artifacts:
- DKD run folder:
	- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/`

Results (validation, final epoch):
- Accuracy: 0.7357
- Macro-F1: 0.7368
- Raw ECE: 0.2119
- Raw NLL: 1.4753
- TS ECE: 0.0348
- TS NLL: 0.7835
- Per-class F1: Angry 0.7436, Disgust 0.6919, Fear 0.7676, Happy 0.7604, Sad 0.7038, Surprise 0.7725, Neutral 0.7176

---

## Experiment: DKD +5 epochs + NegL (resume from KD baseline; NL off)

Date/run stamp:
- 2025-12-29 (`*_20251229_230501`) (completed)

Purpose:
- Test whether NegL helps DKD calibration/per-class behavior when starting from the same KD baseline checkpoint.

NegL config:
- Enabled: yes
- Weight: 0.05
- Ratio: 0.5
- Gate: entropy
- Entropy threshold: 0.7

Command used:
- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 0 -DkdEpochs 5 -DkdResumeFrom outputs\students\KD\mobilenetv3_large_100_img224_seed1337_KD_20251229_182119\checkpoint_last.pt -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7`

Artifacts:
- DKD run folder:
	- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_230501/`

Results (validation, final epoch):
- Accuracy: 0.7351
- Macro-F1: 0.7348
- Raw ECE: 0.2139
- Raw NLL: 1.5033
- TS ECE: 0.0348
- TS NLL: 0.7926
- Per-class F1: Angry 0.7399, Disgust 0.6741, Fear 0.7635, Happy 0.7638, Sad 0.7136, Surprise 0.7688, Neutral 0.7196

### Comparison: DKD baseline vs DKD+NegL

Compare table:
- `outputs/students/_compare_dkd5_negl_vs_dkd5.md`

Quick outcome (from comparison table):
- DKD baseline: acc 0.7357, macro-F1 0.7368, TS ECE 0.0348, TS NLL 0.7835, minority-F1 0.7045
- DKD+NegL: acc 0.7351, macro-F1 0.7348, TS ECE 0.0348, TS NLL 0.7926, minority-F1 0.7024

Interpretation (early, +5 DKD epochs only):
- NegL does not improve the main metrics here; it is slightly worse on macro-F1, TS NLL, and minority-F1.
- This supports the earlier observation that entropy-gated NegL at thresh=0.7 may be too weak/too selective to help consistently.

---

## Experiment: KD 5 epochs — Jan 1, 2026 one-click 3-run sweep (NL vs NegL)

Purpose:
- Make NL and NegL “bite” in isolation (short KD budget) and log whether they stay active across epochs.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_3run_nl_negl_oneclick.ps1 -UseClahe -UseAmp`

Baseline reference:
- KD-only 5ep: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

### Run 1 — NL(proto, penultimate) + fixed threshold (thr=0.05)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_084847/`
- Compare table: `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_fixed_thr0p05_vs_kd5.md`

Results (from compare table):
- acc 0.721527, macro-F1 0.718989
- TS ECE 0.030271, TS NLL 0.807121
- minority-F1 0.686280

NL application stats (from `history.json`):
- applied_frac by epoch: [0.084308, 0.000258, 0.000110, 0.000033, 0.000014]

Interpretation:
- Even with a lower threshold, NL(proto) still becomes nearly inactive after the first epoch.

### Run 2 — NL(proto, penultimate) + top-k gating (frac=0.1)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_091806/`
- Compare table: `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_topk0p1_vs_kd5.md`

Results (from compare table):
- acc 0.723015, macro-F1 0.718769
- TS ECE 0.040034, TS NLL 0.809448
- minority-F1 0.686940

NL application stats (from `history.json`):
- applied_frac by epoch: [0.109375, 0.109375, 0.109375, 0.109375, 0.109375]

Interpretation:
- Top-k gating achieves the behavioral goal: NL remains active every epoch.
- However, this short-run did not show a clear metric gain yet.

### Run 3 — NegL only (entropy gate thr=0.4)

NegL config:
- Weight: 0.05
- Ratio: 0.5
- Gate: entropy
- Entropy threshold: 0.4

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_094542/`
- Compare table: `outputs/students/_compare_20260101_084847_kd5_negl_entropy_ent0p4_vs_kd5.md`

Results (from compare table):
- acc 0.723899, macro-F1 0.720618
- TS ECE 0.039708, TS NLL 0.829301
- minority-F1 0.690973

NegL application stats (from `history.json`):
- applied_frac by epoch: [0.163261, 0.073691, 0.061450, 0.048292, 0.040523]

Interpretation:
- Lowering the entropy threshold from 0.7 → 0.4 makes NegL apply to a meaningful fraction of samples.

### Summary takeaway

- Fixed-threshold NL(proto) still decays to ~0 applied fraction, even with penultimate features.
- Top-k NL gating works as intended (sustained activity).
- NegL with entropy-threshold 0.4 applies much more frequently than 0.7, but metric impact is still mixed in this short budget.

## Experiment: KD 5 epochs — Jan 1, 2026 next-planned one-click sweep

Purpose:
- Execute the “next experiments (planned)” items as a single consistent KD screening batch.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nextplanned_nl_negl_oneclick.ps1 -UseClahe -UseAmp`

Baseline reference:
- KD-only 5ep: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`

### Run 1 — NL-only: NL(proto, penultimate) + top-k gating (frac=0.05, w=0.1)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_153900/`
- Compare table: `outputs/students/_compare_20260101_153859_kd5_nlproto_penultimate_topk0p05_w0p1_vs_kd.md`

Results (from compare table):
- acc 0.727759, macro-F1 0.725666
- TS ECE 0.037482, TS NLL 0.797487
- minority-F1 0.693276

NL application stats (from `history.json` → `nl.applied_frac`):
- applied_frac by epoch: [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

Interpretation:
- Top-k gating successfully holds NL active every epoch at this target fraction.
- Metrics remain close to baseline but slightly worse in this short budget.

### Run 2 — NegL-only: entropy gate (ent=0.3, w=0.05, ratio=0.5)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_165108/`
- Compare table: `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p3_vs_kd.md`

Results (from compare table):
- acc 0.728177, macro-F1 0.726967
- TS ECE 0.046010, TS NLL 0.827339
- minority-F1 0.698288

NegL application stats (from `history.json` → `negl.applied_frac`):
- applied_frac by epoch: [0.227703, 0.127009, 0.109995, 0.086042, 0.066261]

### Run 3 — NegL-only: entropy gate (ent=0.5, w=0.05, ratio=0.5)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_171607/`
- Compare table: `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p5_vs_kd.md`

Results (from compare table):
- acc 0.726782, macro-F1 0.725032
- TS ECE 0.044099, TS NLL 0.824008
- minority-F1 0.690081

NegL application stats (from `history.json` → `negl.applied_frac`):
- applied_frac by epoch: [0.113780, 0.046293, 0.038647, 0.029508, 0.021247]

### Run 4 — Synergy: NL(topk=0.05, w=0.1) + NegL(entropy ent=0.4)

Artifacts:
- Run folder: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20260101_174040/`
- Compare table: `outputs/students/_compare_20260101_153859_kd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_kd.md`

Results (from compare table):
- acc 0.725155, macro-F1 0.722802
- TS ECE 0.042345, TS NLL 0.795800
- minority-F1 0.686232

NL / NegL application stats (from `history.json`):
- NL `applied_frac` by epoch: [0.054688, 0.054688, 0.054688, 0.054688, 0.054688]
- NegL `applied_frac` by epoch: [0.162841, 0.073639, 0.060271, 0.048092, 0.040303]

Interpretation:
- Under this short-run KD budget, the synergy run is worse than baseline on acc/macro-F1 and minority-F1.
- This suggests that (NL top-k at ~5% + NegL entropy 0.4) is not an easy “free win” in this configuration.

## DKD next-planned one-click (status)

Attempted command:
- `scripts/run_nextplanned_nl_negl_oneclick.ps1 -AlsoRunDKD`

Observed issue:
- The DKD one-click failed because `0.5` was mis-bound as `-DkdResumeFrom` (it tried to find a checkpoint at `C:\...\0.5`).

Fix applied (code change):
- Updated the DKD invocation to pass NegL threshold sweep values as a single CSV string argument, and added CSV parsing support in the DKD script.

Next action:
- Re-run the same one-click with `-AlsoRunDKD` to generate the DKD compare markdowns.

Update (Jan 1, 2026):
- DKD resume from KD checkpoint failed due to optimizer state mismatch (`ValueError: parameter group size mismatch`).
- Fix: when resuming across stages (KD -> DKD), `scripts/train_student.py` now resumes **model weights** but skips restoring optimizer/scaler state.
- DKD one-click also no longer parses `Run stamp:` (printed via `Write-Host`, not pipe-capturable); it locates the newest DKD output folder instead.
- A DKD run produced after the fix (output dir): `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_201719/`.

## Experiment: DKD next-planned one-click sweep (Jan 1, 2026)

Purpose:
- Execute the “next-planned” NL/NegL screening set under DKD (resume-from-KD), using the same student/data setup.

One-click command used:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nextplanned_nl_negl_oneclick_dkd.ps1 -UseClahe -UseAmp -NegLEntropyThreshesCsv "0.3,0.5"`

Baseline reference (DKD):
- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/`

### Run 1 — NL-only: NL(proto, penultimate) + top-k gating (frac=0.05, w=0.1)

Artifacts:
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953/`
- Compare: `outputs/students/_compare_20260101_204953_dkd5_nlproto_penultimate_topk0p05_w0p1_vs_dkd.md`

Results (from compare table):
- acc 0.719807, macro-F1 0.717861
- TS ECE 0.045183, TS NLL 0.844715
- minority-F1 0.688264

NL application stats (from `history.json` → `nl.applied_frac`):
- applied_frac by epoch: [0.0625, 0.0625, 0.0625, 0.0625, 0.0625]

### Run 2 — NegL-only: entropy gate (ent=0.3, w=0.05, ratio=0.5)

Artifacts:
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203/`
- Compare: `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p3_vs_dkd.md`

Results (from compare table):
- acc 0.731479, macro-F1 0.730934
- TS ECE 0.041676, TS NLL 0.812235
- minority-F1 0.705310

NegL application stats (from `history.json` → `negl.applied_frac`):
- applied_frac by epoch: [0.088500, 0.059899, 0.041239, 0.031594, 0.028544]

### Run 3 — NegL-only: entropy gate (ent=0.5, w=0.05, ratio=0.5)

Artifacts:
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949/`
- Compare: `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p5_vs_dkd.md`

Results (from compare table):
- acc 0.730410, macro-F1 0.729865
- TS ECE 0.035637, TS NLL 0.805373
- minority-F1 0.703345

NegL application stats (from `history.json` → `negl.applied_frac`):
- applied_frac by epoch: [0.035827, 0.022669, 0.013310, 0.008920, 0.007631]

### Run 4 — Synergy: NL(top-k=0.05, w=0.1) + NegL(entropy ent=0.4)

Artifacts:
- Run dir: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602/`
- Compare: `outputs/students/_compare_20260101_204953_dkd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_dkd.md`

Results (from compare table):
- acc 0.733712, macro-F1 0.733798
- Raw ECE 0.202779, Raw NLL 1.412536
- TS ECE 0.037443, TS NLL 0.786831
- minority-F1 0.701544

NL / NegL application stats (from `history.json`):
- NL `applied_frac` by epoch: [0.054688, 0.054688, 0.054688, 0.054688, 0.054688]
- NegL `applied_frac` by epoch: [0.056472, 0.037731, 0.025051, 0.018169, 0.014909]

### Interpretation (DKD short sweep)

- Under these settings, NL-only (top-k) is not a win: it is materially worse than the DKD baseline on acc/macro-F1 and minority-F1.
- NegL-only at ent=0.3/0.5 applies at a modest rate (few %), but does not improve the main metrics vs the DKD baseline.
- The synergy run improves **raw** calibration/loss (Raw ECE and Raw NLL are lower than baseline) but does not improve acc/macro-F1 or minority-F1; TS metrics are close but not clearly better.

