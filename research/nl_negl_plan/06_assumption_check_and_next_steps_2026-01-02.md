# Assumption Check Report (基於已跑實驗結果) — 2026-01-02

This report checks whether the current assumptions about KD/DKD weaknesses and NL/NegL solutions are supported by **our actual runs** (KD 5ep screening + DKD resume screening).

## 0) Evidence we will use (本報告用嘅 evidence)

Baseline references:
- KD baseline (5ep, NegL/NL off): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119/`
- DKD baseline (as used by compare tables): `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722/`

Key compare markdowns:
- KD (Jan 1, next-planned sweep):
  - `outputs/students/_compare_20260101_153859_kd5_nlproto_penultimate_topk0p05_w0p1_vs_kd.md`
  - `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p3_vs_kd.md`
  - `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p5_vs_kd.md`
  - `outputs/students/_compare_20260101_153859_kd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_kd.md`
- DKD (Jan 1, next-planned DKD one-click):
  - `outputs/students/_compare_20260101_204953_dkd5_nlproto_penultimate_topk0p05_w0p1_vs_dkd.md`
  - `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p3_vs_dkd.md`
  - `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p5_vs_dkd.md`
  - `outputs/students/_compare_20260101_204953_dkd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_dkd.md`

## 1) Background (背景) — your summary

KD (Knowledge Distillation):
- 用 teacher soft targets 去訓練 student
- Use teacher model’s soft targets to train the student

DKD (Decoupled KD):
- 分開 target-class knowledge 同 non-target-class knowledge
- Separate target-class and non-target-class knowledge for more flexible weighting

This background description is consistent with the standard KD/DKD framing.

## 2) Problems Found in KD/DKD (發現問題) — are they supported?

### KD problems

1) Coupled loss structure (loss 耦合)
- Claim: target CE + non-target KL mixed may suppress signal.
- Evidence from our runs: we **did not directly test** different coupling/decoupling forms inside KD here (we only ran a fixed KD recipe).
- Verdict: **Not proven / not disproven** by current experiments.

2) Capacity mismatch / teacher too smooth
- Claim: student underfits due to smooth teacher logits.
- Evidence: our KD baseline and DKD baseline are both reasonably strong; we don’t have a direct “teacher smoothing vs not” ablation.
- Verdict: **Not tested** in current runs.

3) Lack of feature-level guidance
- Claim: logits-only misses spatial/attention alignment.
- Evidence: our NL(proto) uses penultimate features (feature-ish) but the current NL objective is not a spatial/attention alignment loss.
- Verdict: **Directionally true as a concept**, but **not directly validated** by our current metrics.

### DKD problems

1) Still logit-centric
- True by definition for DKD; our pipeline DKD is logit-based.
- Verdict: **Correct description**.

2) Hyperparameter sensitivity
- Claim: α/β/T can destabilize gradients.
- Evidence: we did not run a DKD alpha/beta sweep here; we did see a *resume optimizer state mismatch* bug (engineering issue), not necessarily gradient instability.
- Verdict: **Not tested** as a training-dynamics claim.

3) Teacher calibration issues
- Claim: miscalibrated teacher could amplify error.
- Evidence: we did not vary teacher quality; also our reported TS metrics are on the student evaluation, not teacher.
- Verdict: **Not tested**.

4) Ensemble already strong → extra DKD weight introduces noise
- Evidence: not tested; no DKD weight sweep.
- Verdict: **Not tested**.

## 3) New solutions (新方案) — do NL / NegL help in our data?

### 3.1 NL (Nested Learning) claims vs observed results

Your intended benefits:
- memory module prevents catastrophic forgetting
- difficulty-aware gate dynamically weights loss
- temporal smoothing reduces jitter
- consistency check reduces teacher miscalibration impact

What our NL(proto) actually is in code right now:
- prototype memory + momentum + gating (threshold or top-k), using student penultimate features

What we observed:
- KD short runs: NL(top-k) stays active by construction but did **not** improve metrics vs KD baseline in the 5-epoch budget.
- DKD short runs: NL-only (top-k=0.05, w=0.1) is **clearly worse** than DKD baseline:
  - baseline DKD: acc 0.735711, macro-F1 0.736796, minority-F1 0.704458
  - DKD + NL(top-k): acc 0.719807, macro-F1 0.717861, minority-F1 0.688264

Verdict on NL assumptions (based on our results):
- “NL makes training more stable” → **partially true** (it did not collapse like the legacy learned NegL gate), but **it is not improving accuracy/F1 here**.
- “NL improves minority retention” → **not supported** in these runs (minority-F1 decreased in DKD+NL).
- “NL reduces miscalibration impact” → **not supported** in these runs (TS metrics got worse in DKD+NL).

### 3.2 NegL (Negative Learning) claims vs observed results

Your intended benefits:
- improve calibration (lower ECE)
- sharpen decision boundary / protect minority class
- increase robustness

What we observed:
- KD (5ep): NegL entropy thresholds 0.3/0.5 did not give a clear win; TS metrics tended to worsen.
- DKD (5ep resume):
  - NegL ent=0.3: acc 0.731479, macro-F1 0.730934, minority-F1 0.705310 (very close to baseline; slightly lower acc/F1, slightly higher minority-F1)
  - NegL ent=0.5: acc 0.730410, macro-F1 0.729865, minority-F1 0.703345 (slightly worse than baseline)

Verdict on NegL assumptions (based on our results):
- “NegL improves calibration” → **not consistently supported** (TS ECE/NLL are not clearly better).
- “NegL protects minority class” → **weak/partial evidence** (ent=0.3 gave a tiny minority-F1 increase, but not a broad win).
- “Robustness/domain shift” → **not tested**.

## 4) NL + NegL Synergy (協同效應) — supported?

Your synergy hypothesis:
- NL consistency triggers selective NegL
- class-aware gating
- adaptive thresholds → improve both F1 and ECE

What we actually tested:
- DKD synergy run: NL(top-k=0.05, w=0.1) + NegL(entropy ent=0.4)

Observed DKD synergy results vs DKD baseline:
- baseline DKD: acc 0.735711, macro-F1 0.736796, minority-F1 0.704458, Raw ECE 0.211901, Raw NLL 1.475317
- synergy:      acc 0.733712, macro-F1 0.733798, minority-F1 0.701544, Raw ECE 0.202779, Raw NLL 1.412536

Interpretation:
- Raw calibration/loss improved (Raw ECE and Raw NLL are lower).
- But acc/macro-F1/minority-F1 did not improve.

Verdict on synergy assumption:
- “Improve both F1 + ECE” → **not supported** (ECE improved in raw sense, but F1 did not).
- “Deployment readiness / stability” → **not tested** (needs realtime/online eval, jitter metrics, etc.).

## 5) What your assumptions are missing (最重要缺口)

Based on our actual runs, the biggest gap is:
- We are assuming NL/NegL should be a “free win” on standard metrics, but the data shows they can **hurt** accuracy/F1 under current hyperparams.

So the next step should be framed as:
- not “prove NL/NegL help”, but “find a safe regime (no regressions) then see if calibration/minority can improve”.

## 6) Recommended next steps (下一步建議) — concrete and minimal

A) Validate whether the DKD compare tables are using the intended epoch accounting
- Our DKD one-click is “+5 epochs DKD”, but compare tables report “Epochs = 10”.
- Next: confirm what “Epochs” means in compare output (total vs stage), so the report is interpretable.

B) NL-only: reduce harm first
- Try a smaller NL weight (e.g., 0.02 or 0.05) with the same top-k fraction.
- Rationale: current NL(top-k=0.05, w=0.1) hurts DKD strongly.

C) NegL-only: check if we’re applying enough / too much
- Keep ent=0.3 (it applies more than 0.5) and do a small weight sweep:
  - w = 0.01, 0.02, 0.05 (keep ratio 0.5)
- Goal: see if there is a “sweet spot” that doesn’t worsen macro-F1 while possibly helping calibration.

D) Synergy: only after NL-only stops hurting
- Because NL-only is currently harmful under DKD, synergy is unlikely to be stable.

E) Add a domain-shift test where minority-F1 has room to improve
- Motivation: current in-domain baseline is already strong, so 5ep screening may be underpowered to reveal small gains.
- Target metric: **Minority-F1 (lowest-3)** on the *shifted* domain test split.

Minimal domain-shift protocol (suggested):
- Evaluate existing student checkpoints on **ExpW test** (out-of-domain relative to your mixed/RAF-heavy training).
- Compare DKD baseline vs DKD+NegL(ent=0.3) vs other candidates.

One-click command (runs eval + writes a compare markdown):
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_domain_shift_eval_oneclick.ps1 -EvalManifest Training_data_cleaned/expw_full_manifest.csv -EvalSplit test`

Outputs:
- Eval folders: `outputs/evals/students/<run>__expw_full_manifest__test__<stamp>/`
- Compare table: `outputs/evals/_compare_<stamp>_domainshift_expw_full_manifest_test.md`

How to interpret:
- If a method improves **Minority-F1 (lowest-3)** on ExpW test without large regressions elsewhere, that is stronger evidence than tiny in-domain deltas.
- If results are inconsistent, run 2–3 seeds (same eval protocol) to estimate variance before concluding “no effect”.

---

### Summary (一句講晒)
- 「概念背景」多數正確，但就目前呢組實驗結果：NL/NegL **未證明能提升** F1/accuracy（NL 喺 DKD 仲明顯變差），synergy 亦唔係 free win；下一步應該先搵到 **唔 regress** 嘅超參數區，再講提升 calibration/minority。
