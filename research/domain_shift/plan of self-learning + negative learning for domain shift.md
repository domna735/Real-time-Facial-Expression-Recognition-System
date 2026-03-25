# Plan — Real-time FER domain shift improvement via Self-Learning + Negative Learning

Date: 2026-01-26

## 0) One-sentence goal

Improve real-time FER robustness under webcam/domain shift by adding a **safe, lightweight adaptation loop**: high-confidence pseudo-label self-learning + medium-confidence negative learning safeguards, with rollback.

---

## 1) Feasibility check (can we do this in this repo?)

### 1.1 What already exists (strong support)

- Real-time demo already logs per-frame predictions and probabilities to `demo/outputs/<run_stamp>/per_frame.csv` (and supports CPU forcing).
- Live scorer `scripts/score_live_results.py` already compares `metrics.raw` vs `metrics.smoothed` and reports `macro_f1_present`.
- Student training script `scripts/train_student.py` already includes:
  - standard CE training
  - KD/DKD options
  - **Negative learning scaffold** via `src/fer/negl/losses.py` (`complementary_negative_loss`)
  - NL scaffolding via `src/fer/nl/memory.py` (future extension; not required for this plan)
- Temperature scaling / calibration artifacts exist (teacher/student run dirs have `calibration.json`), enabling meaningful confidence thresholds.

### 1.2 What we still need to implement (new but manageable)

Minimum additional tooling (small scripts) to make the loop real:

1. **Frame capture / buffer builder**: Save selected frames to disk and write a small manifest (CSV) for fine-tuning. Selection is driven by calibrated confidence thresholds and (optionally) stability checks.

1. **Fine-tuning mode**: A “light fine-tune” run that updates only (Option A) final classifier head, or (Option B) BatchNorm stats/affine, or (Option C) last block + head (if A/B too weak). Use conservative hyperparameters and early stopping.

1. **Rollback safety**: Always keep a base checkpoint; only promote the adapted checkpoint if it passes acceptance tests.

### 1.3 Risks and reality check

- Online fine-tuning during live inference is risky on CPU (latency + instability). This plan assumes **offline/periodic fine-tuning** (e.g., after a session).
- Pseudo-label noise can cause collapse; negative learning + gating + rollback are required.

Conclusion: **Possible and realistic** with small incremental work, because logging/scoring/training+NegL scaffolds already exist.

---

## 2) Scope control (what we will / will not do)

### In-scope (MVP)

- Self-learning with high-confidence pseudo-labels (per-user webcam buffer).
- Negative learning for medium-confidence samples (avoid reinforcing wrong pseudo-labels).
- Offline/periodic fine-tuning (small steps) + acceptance tests + rollback.

### Out-of-scope (future work)

- True nested learning / meta-learning inner–outer loops.
- Large-model teacher-on-the-fly during deployment.
- Continual learning across many users with rehearsal strategies.

---

## 3) Definitions and protocol

### 3.1 Confidence thresholds

- $\tau_{high}$: confident pseudo-label threshold (start with 0.90).
- $\tau_{mid}$: medium-confidence band lower bound (start with 0.50).

Policy:

- If $p_{max} \ge \tau_{high}$: treat as pseudo-label **positive** sample.
- If $\tau_{mid} \le p_{max} < \tau_{high}$: treat as **negative-learning** sample (no positive pseudo-label).
- If $p_{max} < \tau_{mid}$: ignore for training (monitor only).

### 3.2 “Stable frame” filter (recommended)

To avoid training on flicker:

- Only accept samples when prediction is stable for N frames (or matches smoothed label), OR when `pred_label != (unstable)`.

### 3.3 What we optimize

Primary (deployment-aligned):

- Live `macro_f1_present` (raw and smoothed)
- Jitter/flip-rate (already computed by live scorer)

Secondary (safety):

- Offline test macro-F1 does not drop beyond a small tolerance (e.g., -0.5% absolute)
- Calibration (ECE/NLL) does not worsen drastically

---

## 4) Acceptance criteria (pass/fail gates)

We only “accept” an adapted checkpoint if it passes all:

1. **Webcam-mini (labeled) improvement**: `macro_f1_present` improves vs baseline OR at least minority classes are not worse.

1. **No major regression on offline eval**: On `Training_data_cleaned/classification_manifest_eval_only.csv` (or existing domain-shift eval suite), macro-F1 drop is within tolerance.

1. **Stability non-worse**: Flip-rate does not increase beyond a set threshold.

If any gate fails: rollback to base checkpoint.

---

## 5) Step-by-step plan (do one by one)

### Step 0 — Establish the baseline (1 day)

Deliverables:

- One labeled live run + `score_results.json` for baseline.
- Domain-shift baseline table already exists; record its key numbers.

Status (done):

- Labeled run folder: `demo/outputs/20260126_205446/`
- Scoring output: `demo/outputs/20260126_205446/score_results.json`
- Baseline (protocol-scored):
  - `raw.macro_f1_present` = 0.4721, `raw.accuracy` = 0.5284
  - `smoothed.macro_f1_present` = 0.5248, `smoothed.accuracy` = 0.5879
  - Biggest weak classes in this run: `Fear` F1 = 0.0, `Sad` F1 ≈ 0.03

Commands (live baseline):

```powershell
python demo/realtime_demo.py --model-kind student --device cpu
python scripts/score_live_results.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --out demo/outputs/<run_stamp>/score_results.json --pred-source both
```

### Step 1 — Create “webcam-mini” evaluation set (0.5–1 day)

Goal: a small, repeatable labeled set (even 2–5 minutes split into 2–3 clips) used as the main deployment KPI.

Deliverables:

- Stored clip(s) + manual labels protocol documented.

### Step 2 — Build the self-learning buffer dataset (1–2 days)

Input:

- `demo/outputs/<run_stamp>/per_frame.csv`

Output:

- A folder of selected frame images
- A manifest CSV (same label schema as training; canonical 7 labels)

Implementation (done for manual-labeled buffer):

- Buffer builder script: `scripts/build_webcam_buffer.py`
- Example output from the first run:
  - `demo/outputs/20260126_205446/buffer_manual/images/` (426 images)
  - `demo/outputs/20260126_205446/buffer_manual/manifest.csv`

Command (manual labels + face crops):

```powershell
python scripts/build_webcam_buffer.py --per-frame demo/outputs/20260126_205446/per_frame.csv --video demo/outputs/20260126_205446/session_annotated.mp4 --out-dir demo/outputs/20260126_205446/buffer_manual --min-frame-gap 10 --max-per-class 250 --face-crop
```

Implementation (new for pseudo-label self-learning buffer):

- Buffer builder script: `scripts/build_webcam_selflearn_buffer.py`
- Output includes two extra columns in `manifest.csv`:
  - `weight`: per-sample CE weight (1 for high-confidence pseudo-labels, 0 for NegL-only samples)
  - `neg_label`: explicit NegL target label (filled only for NegL-only samples)

Command (pseudo-labels + NegL-only band + face crops):

```powershell
python scripts/build_webcam_selflearn_buffer.py --per-frame demo/outputs/20260126_205446/per_frame.csv --video demo/outputs/20260126_205446/session_annotated.mp4 --out-dir demo/outputs/20260126_205446/buffer_selflearn --tau-high 0.90 --tau-mid 0.50 --stable-rule raw_eq_smoothed --require-probs --require-not-unstable --face-crop
```

Selection policy (start simple):

- Keep frames with $p_{max} \ge \tau_{high}$ and stable prediction
- Cap per-class samples to avoid overfitting to one emotion (e.g., max 200 per class per session)

### Step 3 — Self-learning fine-tune (MVP) (1–2 days)

Goal: small-step update to improve webcam-mini without harming offline.

Variants (do in order):

1) Head-only fine-tune
2) BN-only update
3) Last block + head (only if needed)

Conservative settings:

- Very small LR (e.g., 1e-5 to 3e-5)
- Few epochs (e.g., 1–5)
- Strong regularization / early stop

Deliverables:

- Adapted checkpoint + metrics JSON
- Before/after comparison on webcam-mini and offline eval

Status (partially done):

- Head-only fine-tune completed:
  - Output: `outputs/students/FT_webcam_head_20260126_1/`
  - Checkpoint: `outputs/students/FT_webcam_head_20260126_1/best.pt`
- Offline regression gate (eval-only) was checked and failed for this first attempt:
  - Baseline macro-F1=0.4859 (CE20251223)
  - Adapted macro-F1=0.4508 (FT_webcam_head_20260126_1)
- New labeled webcam-mini run recorded and scored:
  - Run: `demo/outputs/20260126_215903/`
  - Smoothed macro-F1_present=0.5552 (baseline run smoothed was 0.5248)

- BN-only fine-tune completed (1st attempt):
  - Output: `outputs/students/FT_webcam_bn_20260126_1/`
  - Checkpoint: `outputs/students/FT_webcam_bn_20260126_1/best.pt`
  - Offline eval-only regression metrics:
    - accuracy=0.5486, macro-F1=0.4513
    - TS ECE=0.0606
  - Interpretation: offline gate still fails at this setting; need either a smaller update (lower LR) and/or tighter/cleaner buffer sampling.

Interpretation (why it helped / why it got worse):

What we can say confidently from the current evidence:

- The adapted checkpoint shows **better target-domain behavior** on our webcam scoring protocol (especially for previously weak classes like Fear/Sad in the second run).
- The adapted checkpoint shows a **clear offline regression** on the eval-only manifest, meaning it is not safe to promote yet.

Why webcam-mini improved (most likely mechanisms):

- **Domain shift is real**: the webcam pipeline (sensor, lighting, pose/scale, background, compression) differs from the training sources; adapting using target-domain crops can reduce that gap.
- **The deployment metric is smoothed**: a small improvement in stability/consistency of logits can yield a larger improvement after EMA/vote/hysteresis smoothing.

Why offline eval-only got worse (most likely mechanisms):

- **Small, correlated buffer**: the buffer is one subject/session; gradients can push the model to specialize to that “style”, moving decision boundaries away from the multi-source distribution.
- **Frame-level label noise**: even with manual labeling, some frames are ambiguous/transition frames; face detection/crops can fail or include partial faces.
- **Distribution mismatch + imbalance**: even with per-class caps, the webcam dataset has much narrower variation than `Training_data_cleaned`.

Important caution:

- A new webcam run is not identical to the baseline run (emotion mix / duration / lighting). Improvement is encouraging but should be confirmed with repeat runs.
- This is why the acceptance gate requires both webcam-mini improvement and offline non-regression.

Decision (current):

- Keep `outputs/students/FT_webcam_head_20260126_1/` as an experiment artifact only.
- Do not promote it for deployment until both gates pass.

Command used (head-only):

```powershell
python scripts/train_student.py --mode ce --manifest demo/outputs/20260126_205446/buffer_manual/manifest.csv --data-root . --init-from outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt --tune head --epochs 3 --batch-size 64 --num-workers 0 --lr 3e-5 --weight-decay 0.0 --warmup-epochs 0 --output-dir outputs/students/FT_webcam_head_20260126_1
```

Command example (self-learning + NegL buffer, BN-only):

```powershell
python scripts/train_student.py --mode ce --manifest demo/outputs/20260126_205446/buffer_selflearn/manifest.csv --data-root demo/outputs/20260126_205446/buffer_selflearn --init-from outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt --tune bn --epochs 1 --batch-size 64 --num-workers 0 --lr 1e-5 --weight-decay 0.0 --warmup-epochs 0 --use-negl --manifest-use-weights --manifest-use-neg-label --negl-weight 0.1 --negl-ratio 1.0 --negl-gate none --output-dir outputs/students/FT_webcam_selflearn_bn_20260126_1
```

Command used (offline eval-only regression check):

```powershell
python scripts/eval_student_checkpoint.py --checkpoint outputs/students/FT_webcam_head_20260126_1/best.pt --eval-manifest Training_data_cleaned/classification_manifest_eval_only.csv --eval-data-root Training_data_cleaned --eval-split test --batch-size 256 --num-workers 0
```

Next (BN-only adaptation, safer update):

Rationale:

- BN parameters/statistics often encode assumptions about feature distribution driven by image appearance (brightness/contrast/color). Updating BN can adapt to webcam lighting/capture style.
- BN-only updates are typically **less likely** to destroy class semantics than updating classifier weights (head) on a small target buffer.

Proposed conservative recipe (start smallest first):

- `--tune bn`
- `--epochs 1` (only increase to 2 if it is stable)
- `--lr 1e-5` (only increase if it underfits)
- Keep `--weight-decay 0.0`, `--warmup-epochs 0`

Command (BN-only):

```powershell
python scripts/train_student.py --mode ce --manifest demo/outputs/20260126_205446/buffer_manual/manifest.csv --data-root . --init-from outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt --tune bn --epochs 1 --batch-size 64 --num-workers 0 --lr 1e-5 --weight-decay 0.0 --warmup-epochs 0 --output-dir outputs/students/FT_webcam_bn_20260126_1
```

Then repeat the same two acceptance gates:

1. Offline eval-only regression check (must not meaningfully drop vs baseline):

```powershell
python scripts/eval_student_checkpoint.py --checkpoint outputs/students/FT_webcam_bn_20260126_1/best.pt --eval-manifest Training_data_cleaned/classification_manifest_eval_only.csv --eval-data-root Training_data_cleaned --eval-split test --batch-size 256 --num-workers 0
```

1. Webcam-mini improvement check (new labeled run + same scoring protocol):

```powershell
python demo/realtime_demo.py --model-kind student --device cpu --model-path outputs/students/FT_webcam_bn_20260126_1/best.pt --record-video --record-video-mode annotated
python scripts/score_live_results.py --per-frame demo/outputs/<new_run_stamp>/per_frame.csv --out demo/outputs/<new_run_stamp>/score_results.json --pred-source both
```

If either gate fails, reduce update size further:

- Lower LR to `5e-6`, keep `--epochs 1`
- Reduce buffer size per class (smaller `--max-per-class`)
- Tighten frame selection (`--min-frame-gap`, `--stable-only`, keep `--face-crop`)

### Step 4 — Add negative learning for medium-confidence samples (1–3 days)

Purpose: reduce pseudo-label collapse by using medium-confidence frames as “NOT-this-class” signals.

Implementation approach (match existing code):

- Use `src/fer/negl/losses.py` (`complementary_negative_loss`) to penalize probability mass on sampled negative classes.

How to choose negative classes (start simple):

- For a medium-confidence sample, pick 1–3 classes with **lowest predicted probabilities** as negatives (safe), OR
- Avoid only one “hard confusable” class at first to not hurt recall.

Loss mix:

- High-confidence samples: CE on pseudo-label
- Medium-confidence samples: NegL only (no positive label)

Deliverables:

- Ablation table: Self-learning only vs Self+NegL

### Step 5 — Threshold + ablation sweep (2–4 days)

Run a small controlled grid (keep it small):

- $\tau_{high}$ in {0.85, 0.90, 0.95}
- $\tau_{mid}$ in {0.40, 0.50, 0.60}
- Update mode: head-only vs BN-only
- Buffer size cap

Deliverables:

- One markdown table summarizing gains/trade-offs

### Step 6 — Write-up + positioning (1–2 days)

Deliverables:

- A short methodology section for report: motivation, loop, safeguards, evaluation protocol.
- A failure analysis section: when self-learning helps, when it hurts, how NegL prevents collapse.

---

## 6) Practical notes (what to discuss with supervisor)

Key decisions to confirm:

1) Deployment KPI priority: webcam-mini improvement vs ExpW robustness.
2) Whether we are allowed to use user data (webcam frames) for adaptation (privacy/storage policy).
3) Acceptance thresholds (FPS/latency + accuracy/F1 + flip-rate).
4) Whether calibration is a required output for deployment (confidence gating / “unknown”).

---

## 7) Optional extension (future work, low-risk to mention)

- Nested learning: meta-learn the update rule using multi-source domains (RAFDB → AffectNet → ExpW).
- Large teacher guidance: periodically label keyframes using a stronger teacher, then KD to student.

---

## 8) 中文摘要（俾教授／報告用，繁體）

目標：用「自學（pseudo-label self-learning）」+「負學習（negative learning）」做一個安全、可 rollback 的 adaptation loop，令 real-time webcam domain shift 下的 FER 表現更穩定、更準。

核心做法：

- 高信心（$p_{max} \ge \tau_{high}$）：當作 pseudo-label 正樣本，加入 buffer 做小步 fine-tune。
- 中信心（$\tau_{mid} \le p_{max} < \tau_{high}$）：唔強行 assign 正 label，用負學習學「唔係某啲 class」，減少錯 pseudo-label 放大。
- 低信心（$p_{max} < \tau_{mid}$）：唔用嚟訓練，只監察。

安全機制：

- 只做 offline/periodic fine-tune（避免即時 inference 延遲）
- 小 LR + 少 epochs + 只更新 head/BN（防止 catastrophic drift）
- 必須做 acceptance test（webcam-mini + offline eval），唔 pass 就 rollback。

## 9) Executive Summary

This work proposes a safe, lightweight adaptation loop for real-time FER under webcam domain shift.
The system integrates high-confidence pseudo-label self-learning with medium-confidence negative learning, combined with strict acceptance gates and rollback mechanisms.
Initial experiments show that target-domain adaptation improves live webcam performance, especially on minority classes, but naïve head-only updates cause offline regression.
This confirms the need for safer update policies such as BN-only adaptation and tighter buffer sampling.
The pipeline is fully reproducible and aligns with deployment constraints (CPU, real-time, smoothed metrics).
