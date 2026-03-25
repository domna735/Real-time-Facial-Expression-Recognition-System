# Process Log - Week 5 of December
This document captures the daily activities, decisions, and reflections during the fifth week of December 2025, focusing on reconstructing the facial expression recognition system as per the established plan.

## 2025-12-28 | KD 5-epoch ablation with NegL (entropy gate)
Intent:
- Run a short, fair (5-epoch) KD ablation with NegL enabled to validate wiring, logging, and check early effects on calibration/per-class performance.

Action:
- Ran student runner with CE skipped and NegL enabled:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7 -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 2 -UseClahe -UseAmp`
- Output folders created:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720`
	- Logs: `outputs/students/Log and test/_logs_20251228_233720/`

Result:
- KD (+NegL) 5 epochs completed with stable Windows settings.
- Validation (final epoch) from `reliabilitymetrics.json`:
	- Accuracy: 0.7224
	- Macro-F1: 0.7198
	- Raw NLL: 1.7801
	- Raw ECE: 0.2106
	- Temperature-scaled (global T=4.3918): NLL 0.8085, ECE 0.0398
	- Per-class F1 (final epoch): Angry 0.7130, Disgust 0.6317, Fear 0.7568, Happy 0.7550, Sad 0.7036, Surprise 0.7646, Neutral 0.7139
- NegL logging (from `history.json`): entropy gate was very selective.
	- Applied fraction dropped from ~3.20% (epoch 0) to ~0.58% (epoch 4).
	- Mean entropy dropped from 0.282 (epoch 0) to 0.081 (epoch 4).

Notes / Issues observed:
- CE stage used `-CeEpochs 0`, so CE stage is skipped (expected).
- DKD stage used `-DkdEpochs 0`, so DKD stage is skipped (expected).

Decision / Interpretation:
- This run validates NegL wiring + logging and provides a KD(+NegL) 5-epoch reference point.
- Because the entropy threshold is high (0.7) and entropy is typically low, NegL was applied to a very small subset of samples; any NegL effect is likely muted at these settings.

Next:
- Run baseline KD-only 5 epochs with identical settings but NegL off, then compare (completed on 2025-12-29; see entry below).
- If NegL effect looks promising, re-run DKD with a positive additional epoch budget (e.g., `-DkdEpochs 5`) and/or lower the entropy threshold so NegL applies more often.

## 2025-12-29 | KD-only 5-epoch fair baseline (NegL off) + comparison table
Intent:
- Produce a fair baseline to compare against the existing KD+NegL 5-epoch run (same epochs, same batch size, same data/softlabels, NegL disabled).

Action:
- Ran KD-only with CE/DKD skipped:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp`
- Baseline output:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119`
	- Logs: `outputs/students/Log and test/_logs_20251229_182119/`

Result:
- KD-only 5 epochs completed.
- Validation (final epoch) from `reliabilitymetrics.json`:
	- Accuracy: 0.7284
	- Macro-F1: 0.7266
	- Raw NLL: 1.7429
	- Raw ECE: 0.2130
	- Temperature-scaled: NLL 0.7839, ECE 0.0271
	- Minority-F1 (lowest-3 classes): 0.6973

Comparison:
- Generated a 2-run comparison markdown:
	- `outputs/students/_compare_kd5_vs_negl5.md`
- Compared runs:
	- KD-only (5ep): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119`
	- KD+NegL (5ep): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720`

Interpretation (early, 5 epochs only):
- KD-only is slightly higher on accuracy/macro-F1 and slightly lower NLL vs KD+NegL at these settings.
- Temperature-scaled ECE is better for KD-only in this snapshot (0.0271 vs 0.0398).

Next:
- Proceed to the next planned ablations (keeping epoch budget fixed per stage): KD+NegL+NL, then DKD+NegL, then DKD+NegL+NL.

## 2025-12-29 | KD 5-epoch ablation with NegL + NL gate (completed)
Intent:
- Test the “KD+NegL+NL” variant after establishing the KD-only baseline and KD+NegL reference.
- Keep the epoch budget fixed (5 epochs) for fair early-stage comparison.

Action:
- Started KD stage only (CE/DKD skipped) with NegL enabled and NL enabled:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7 -UseNL -NLHiddenDim 32 -NLLayers 1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_negl_nl_vs_kd5.md`
- Output folder (planned by runner):
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_194408`
- Logs:
	- `outputs/students/Log and test/_logs_20251229_194408/`

Note:
- A prior run `*_20251229_191103` was started before the runner correctly passed `--use-nl`, so it is KD+NegL only (no NL). Use `*_20251229_194408` for the real KD+NegL+NL result.

Result:
- KD+NegL+NL 5 epochs completed.
- Validation (final epoch) from `reliabilitymetrics.json`:
	- Accuracy: 0.5402
	- Macro-F1: 0.5204
	- Raw NLL: 1.5776
	- Raw ECE: 0.2349
	- Temperature-scaled (global T=2.4015): NLL 1.2021, ECE 0.0315
	- Per-class F1 (final epoch): Angry 0.1961, Disgust 0.6371, Fear 0.6985, Happy 0.6399, Sad 0.4970, Surprise 0.6178, Neutral 0.3563
- Auto-compare generated:
	- `outputs/students/_compare_kd5_negl_nl_vs_kd5.md`
	- Compared vs KD-only baseline `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119`

Decision / Interpretation:
- The NL wiring is confirmed working (flags passed and NL status appears in the compare table), but the current NL gate configuration is **not safe**: it causes a large regression vs KD-only baseline.
- Therefore, NL should be treated as a “failed ablation” under the current settings, and we should avoid stacking it into DKD until stabilized.

Next:
- Proceed to **DKD + NegL (without NL)** as the next planned ablation step.
- When returning to NL debugging, first reduce NegL pressure (e.g., lower `-NegLWeight`/`-NegLRatio`) or temporarily disable entropy gating (`-NegLGate none`) to isolate the unstable interaction.

## 2025-12-29 | DKD +5 epochs resumed from KD baseline (NegL off)
Intent:
- Create a DKD baseline that is directly comparable to DKD+NegL by resuming from the same KD checkpoint (no re-running KD).

Action:
- Ran DKD-only for +5 epochs, resuming from KD baseline `*_20251229_182119`:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 0 -DkdEpochs 5 -DkdResumeFrom outputs\students\KD\mobilenetv3_large_100_img224_seed1337_KD_20251229_182119\checkpoint_last.pt -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp`
- Output folder:
	- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722`

Result:
- DKD resumed at `start_epoch=5` and ran epochs 5..9 (total epochs=10 in history).
- Validation (final epoch) from `reliabilitymetrics.json`:
	- Accuracy: 0.7357
	- Macro-F1: 0.7368
	- Raw NLL: 1.4753
	- Raw ECE: 0.2119
	- Temperature-scaled (global T=3.1541): NLL 0.7835, ECE 0.0348
	- Minority-F1 (lowest-3 classes): 0.7045

Decision / Interpretation:
- This is a clean DKD baseline anchored to the KD-only checkpoint, suitable for fair DKD+NegL comparison.

Next:
- Run the matching DKD+NegL variant (completed in the next entry).

## 2025-12-29 | DKD +5 epochs resumed from KD baseline (NegL on; NL off)
Intent:
- Test whether NegL helps DKD when starting from the same KD baseline checkpoint.

Action:
- Ran DKD + NegL for +5 epochs, resuming from the same KD baseline checkpoint:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 0 -DkdEpochs 5 -DkdResumeFrom outputs\students\KD\mobilenetv3_large_100_img224_seed1337_KD_20251229_182119\checkpoint_last.pt -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7`
- Output folder:
	- `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_230501`

Result:
- Validation (final epoch) from `reliabilitymetrics.json`:
	- Accuracy: 0.7351
	- Macro-F1: 0.7348
	- Raw NLL: 1.5033
	- Raw ECE: 0.2139
	- Temperature-scaled (global T=3.1825): NLL 0.7926, ECE 0.0348
	- Minority-F1 (lowest-3 classes): 0.7024

Comparison:
- Generated compare markdown:
	- `outputs/students/_compare_dkd5_negl_vs_dkd5.md`
- Summary:
	- DKD baseline: acc 0.7357, macro-F1 0.7368, TS ECE 0.0348, TS NLL 0.7835
	- DKD+NegL: acc 0.7351, macro-F1 0.7348, TS ECE 0.0348, TS NLL 0.7926

Decision / Interpretation:
- Under these settings (entropy gate thresh=0.7, w=0.05, ratio=0.5), NegL does not improve DKD metrics; it is slightly worse on macro-F1 / TS NLL / minority-F1.

Next:
- If we continue NegL work, do a gate/strength sweep (lower entropy threshold and/or lower weight/ratio) rather than proceeding with the current default.
- Keep NL out of DKD until the NL mechanism is stabilized in KD-only experiments.

## 2025-12-30 | KD 5-epoch ablation with NL(proto) (NegL off) + comparison
Intent:
- Stabilize NL by testing the new NL(proto) mechanism **without** NegL under the same 5-epoch KD budget.

Action:
- Ran KD stage only (CE/DKD skipped) with NL(proto) enabled:
	- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.2 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_vs_kd5.md`
- Output folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251230_004048`
- Logs:
	- `outputs/students/Log and test/_logs_20251230_004048/`

Result:
- KD+NL(proto) 5 epochs completed without instability.
- From compare table `outputs/students/_compare_kd5_nlproto_vs_kd5.md`:
	- KD-only baseline: acc 0.728363, macro-F1 0.726648, TS ECE 0.027051, TS NLL 0.783856, minority-F1 0.697342
	- KD+NL(proto): acc 0.729573, macro-F1 0.728076, TS ECE 0.042676, TS NLL 0.796150, minority-F1 0.694379
- NL(proto) stats (final epoch, from `history.json`):
	- train_nl_loss ~ 1.68e-05
	- applied_frac ~ 7.16e-05
	- sim_mean ~ 0.975

Decision / Interpretation:
- This is a **stable** NL variant (no collapse), but it is currently too selective: with `NLConsistencyThresh=0.2`, NL is applied to ~0.007% of samples.
- Next tweak should target making NL active while staying safe:
	- Lower `-NLConsistencyThresh` (e.g., 0.05–0.10) and re-run the same KD 5-epoch test before combining with NegL.

## 2025-12-31 | NL(proto) consistency-threshold sweep (NegL off)

Intent:
- Make NL(proto) “bite” (increase `applied_frac`) while keeping KD stable, before moving on to NegL integration.

Action:
- Ran 5-epoch KD-only (CE/DKD skipped) with NL(proto), sweeping the consistency threshold:
	- thr=0.10:
		- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.10 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p10_vs_kd5.md`
		- Output: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_031407/`
		- Compare: `outputs/students/_compare_kd5_nlproto_thr0p10_vs_kd5.md`
	- thr=0.05:
		- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.05 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p05_vs_kd5.md`
		- Output: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_071347/`
		- Compare: `outputs/students/_compare_kd5_nlproto_thr0p05_vs_kd5.md`
	- thr=0.005:
		- `powershell -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 128 -NumWorkers 8 -UseClahe -UseAmp -UseNL -NLKind proto -NLDim 32 -NLMomentum 0.9 -NLConsistencyThresh 0.005 -NLWeight 0.1 -CompareWith outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119 -CompareOut outputs/students/_compare_kd5_nlproto_thr0p005_vs_kd5.md`
		- Output: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_074714/`
		- Compare: `outputs/students/_compare_kd5_nlproto_thr0p005_vs_kd5.md`

Result:
- From compare tables:
	- thr=0.10: acc 0.731944, macro-F1 0.730627, TS ECE 0.032719, TS NLL 0.762842, minority-F1 0.698116
	- thr=0.05: acc 0.727061, macro-F1 0.724867, TS ECE 0.022909, TS NLL 0.762270, minority-F1 0.691834
	- thr=0.005: acc 0.732223, macro-F1 0.731533, TS ECE 0.030852, TS NLL 0.763651, minority-F1 0.702075
- NL(proto) stats (final epoch, from `history.json` → `nl` object):
	- thr=0.10: train_nl_loss 1.70e-05, applied_frac 1.38e-04 (~0.0138%), sim_mean 0.9883
	- thr=0.05: train_nl_loss 1.61e-05, applied_frac 2.29e-04 (~0.0229%), sim_mean 0.9948
	- thr=0.005: train_nl_loss 1.98e-06, applied_frac 2.77e-04 (~0.0277%), sim_mean 0.9995

Decision / Interpretation:
- All three sweeps remain **almost inactive** (applied_frac stays <0.03%).
- The cosine similarity is extremely high by the end of training (sim_mean ~0.99–0.9995), so the current consistency gating rarely triggers.
- This suggests the current proto representation (projected from student logits) is too “easy” to align, so NL does not deliver a meaningful training signal.

Next:
- Keep NegL off for now.
- To make NL meaningful, change NL(proto) to use a richer embedding source (e.g., student penultimate features) rather than logits, then repeat the same KD 5-epoch test and re-check `applied_frac`.

## 2025-12-31 | NL(proto) switched to penultimate embedding + KD 5-epoch check

Intent:
- Implement the planned fix: use student penultimate features for NL(proto) embeddings (instead of logits) so the consistency gate becomes non-degenerate.
- Re-run the same 5-epoch KD-only ablation and verify `applied_frac` becomes non-trivial.

Action:
- Added `--nl-embed {penultimate,logits}` for NL(proto) (default: penultimate) and updated the runner to pass `-NLEmbed`.
- Ran KD stage only (CE/DKD skipped) with NL(proto) enabled using penultimate embeddings:
	- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0 -BatchSize 64 -NumWorkers 8 -UseNL -NLKind proto -NLEmbed penultimate -NLConsistencyThresh 0.2 -NLWeight 0.1`
- Output folder:
	- `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251231_155841/`
- Logs:
	- `outputs/students/Log and test/_logs_20251231_155841/`
- Generated compare markdown:
	- `outputs/students/_compare_kd5_nlproto_penultimate_thr0p2_vs_kd5.md`

Result:
- NL(proto) is now clearly active early in training (from `history.json`):
	- applied_frac by epoch: [0.041732, 0.000138, 0.000033, 0.000010, 0.000000]
	- sim_mean (final epoch): 0.9860
- From compare table vs KD-only baseline `*_20251229_182119`:
	- KD-only (5ep): acc 0.728363, macro-F1 0.726648, TS ECE 0.027051, TS NLL 0.783856, minority-F1 0.697342
	- KD+NL(proto, penultimate, thr=0.2) (5ep): acc 0.726689, macro-F1 0.724393, TS ECE 0.039511, TS NLL 0.799723, minority-F1 0.691421

Decision / Interpretation:
- The key hypothesis is confirmed: switching from logits -> penultimate features makes the NL(proto) gate non-degenerate.
- But the NL signal decays to ~0 after the first epoch, suggesting prototypes align very quickly and the fixed consistency threshold stops triggering.

Next:
- Stay in NL-only mode (NegL off) and tune for a steadier applied fraction across epochs (e.g., lower `-NLConsistencyThresh`, reduce momentum, or adjust NL weight), then re-run the same KD 5-epoch check.