# Domain Shift Improvement via Self-Learning + Negative Learning (NegL)

Date: 2026-01-28 (updated: 2026-02-05)

Primary plan document:

- [plan of self-learning + negative learning for domain shift.md](../plan%20of%20self-learning%20%2B%20negative%20learning%20for%20domain%20shift.md)

This report is an evidence-backed summary of what we have actually run in this repo for the webcam/domain-shift track.

Important note (scope):

- Sections labeled **Executed (evidence)** are strictly backed by artifacts listed in this report.
- Sections labeled **Planned (design)** describe what we intend to do next, and are not claimed as results.

## Feb 2026 addendum (2026-02-21): safety-gated A/B update (executed evidence)

After the 2026-02-05 update, we executed a first conservative Self-Learning + manifest-driven NegL adaptation attempt with an explicit offline gate + same-session webcam replay A/B.

Canonical write-up (kept in the final report for consolidation):

- Final report Section 4.9.2 “2026-02-21 update”: `research/final report/final report version 2.md`

Evidence artifacts:

- Webcam same-session A/B:
   - Baseline: `demo/outputs/20260126_205446/score_results.json`
   - Adapted: `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`
- Offline eval-only gate:
   - Baseline: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/baseline/`
   - Adapted: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/adapted_clahe_head_frozebn/`

Key conclusion from this addendum:

- The offline gate can be passed when preprocessing and BatchNorm behavior are controlled, but the first conservative adapted checkpoint still regressed on the deployment-facing same-session webcam score.

## 0) Evidence inventory (where the numbers come from)

### Live (webcam) scoring artifacts

- Baseline labeled run: `demo/outputs/20260126_205446/score_results.json`
- Later labeled run: `demo/outputs/20260126_215903/score_results.json`

These contain:

- `metrics.raw` vs `metrics.smoothed` (macro-F1, accuracy, per-class F1, supports)
- `jitter_flips_per_min`

### Offline regression gate artifacts (eval-only)

Eval-only manifest:

- `Training_data_cleaned/classification_manifest_eval_only.csv`

Saved results (per run) under:

- `outputs/evals/students/<run>/reliabilitymetrics.json`

Runs we verified:

- Baseline CE: `outputs/evals/students/_baseline_CE20251223_eval_only_test/reliabilitymetrics.json`
- Head-only fine-tune: `outputs/evals/students/FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358/reliabilitymetrics.json`
- BN-only fine-tune: `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/reliabilitymetrics.json`

## 1) Live results (webcam labeled runs) — Executed (evidence)

Important: the two labeled runs have different emotion mix and duration (see per-class supports in each JSON). We treat these as *evidence of behavior*, not a final A/B conclusion.

### 1.1 Baseline run (20260126_205446)

Source: `demo/outputs/20260126_205446/score_results.json`

- Raw: accuracy=0.5284, macro_f1_present=0.4721
- Smoothed: accuracy=0.5879, macro_f1_present=0.5248
- Minority-F1 (lowest-3): raw=0.1562, smoothed=0.1609
- Jitter flips/min: 14.8644

Notable per-class F1 (smoothed):

- Fear=0.0000 (support=393)
- Sad=0.0301 (support=542)

### 1.2 Later labeled run (20260126_215903)

Source: `demo/outputs/20260126_215903/score_results.json`

- Raw: accuracy=0.4638, macro_f1_present=0.4933
- Smoothed: accuracy=0.5139, macro_f1_present=0.5552
- Minority-F1 (lowest-3): raw=0.3369, smoothed=0.3618
- Jitter flips/min: 17.9104

Notable per-class F1 (smoothed):

- Fear=0.3432 (support=1316)
- Sad=0.4372 (support=1540)

Interpretation (evidence-based):

- Compared to the baseline run, this later run shows much higher Fear/Sad supports and much higher Fear/Sad F1.
- Jitter is higher in this run; improvements in smoothed macro-F1 do not necessarily imply “more stable” behavior.

## 2) Offline eval-only regression gate (safety)

This is the gate that decides whether we can promote an adapted checkpoint beyond “experiment artifact”.

### 2.1 Baseline (CE20251223)

Source: `outputs/evals/students/_baseline_CE20251223_eval_only_test/reliabilitymetrics.json`

- Raw: accuracy=0.5674, macro-F1=0.4859
- TS: ECE=0.0592, NLL=1.2288

### 2.2 Head-only fine-tune (FT_webcam_head_20260126_1)

Source: `outputs/evals/students/FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358/reliabilitymetrics.json`

- Raw: accuracy=0.5480, macro-F1=0.4508
- TS: ECE=0.0600, NLL=1.2894

Decision:

- Offline gate fails (macro-F1 drops from 0.4859 → 0.4508).

### 2.3 BN-only fine-tune (FT_webcam_bn_20260126_1)

Source: `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/reliabilitymetrics.json`

- Raw: accuracy=0.5486, macro-F1=0.4513
- TS: ECE=0.0606, NLL=1.2890

Decision:

- Offline gate fails (macro-F1 drops from 0.4859 → 0.4513).

## 3) Why we did these steps (and why results are mixed)

### 3.1 Why do self-learning / adaptation at all?

Because the deployment domain (webcam) differs from training distributions. The live scoring logs show that certain classes (e.g., Fear/Sad) can be weak under one webcam condition/run, which is consistent with domain shift.

### 3.2 Why can webcam metrics improve while offline gets worse?

Most likely contributors (consistent with our current artifacts):

- Small correlated buffer: our adaptation buffer is built from one session/subject, so it can push the model toward that style.
- Ambiguous/transition frames: even with manual labels, some frames are borderline; training on these can move boundaries in unhelpful ways.
- Distribution mismatch: offline eval-only covers broader sources; a tiny webcam buffer is not representative.

### 3.3 Why did BN-only not “save” offline performance here?

BN-only is often safer, but it can still shift feature statistics enough to hurt generalization if the buffer is narrow or if the update (even 1 epoch) is still too large for the target signal quality.

---

## 4) What Self-Learning and NegL mean in this repo — Planned (design)

This section explains *how* Self-Learning and Negative Learning are intended to be used for webcam domain shift, and where they map into existing code.

### 4.1 Self-Learning: what it is and why it can help

Self-Learning here means: use the current model to generate **pseudo-labels** on target-domain webcam frames, then fine-tune the student on those pseudo-labeled crops.

Why it can help under webcam shift:

- The student sees target-domain appearance (camera noise, lighting, blur, crop style), so feature statistics can move toward the deployment domain.
- If we select only high-confidence and stable frames, we can get a “clean enough” signal that improves robustness without fully overfitting.

Where it fits in the repo:

- Data source: `demo/outputs/<run_stamp>/per_frame.csv`
- Buffer builder: `scripts/build_webcam_buffer.py` (already exists)
- Fine-tune entrypoint: `scripts/train_student.py` (already supports `--init-from` and tuning scopes)

### 4.2 Negative Learning (NegL): what it is and why it helps

NegL in this repo is implemented as a **complementary-label negative learning** auxiliary loss:

- Code: `src/fer/negl/losses.py` (`complementary_negative_loss`)
- Training integration: `scripts/train_student.py` flags `--use-negl`, `--negl-weight`, `--negl-ratio`, and entropy gating (`--negl-gate entropy --negl-entropy-thresh ...`).

Conceptually, NegL helps self-learning by reducing “confirmation bias”:

- Self-learning can reinforce wrong pseudo-labels when the model is uncertain.
- NegL is intended to be applied on **medium-confidence / high-entropy** samples to discourage overconfident mistakes (push probability mass away from sampled wrong classes).

Implementation detail (current code behavior):

- NegL currently samples a wrong class **uniformly** among classes not equal to the (current) label `y` in the batch.
- This is a valid scaffold, but a future improvement is **teacher-guided** or **confusion-guided** negative sampling (not yet implemented).

### 4.3 Confidence-banded policy (the intended Self-Learning + NegL loop)

We use a 3-band rule based on calibrated confidence (from per-frame probabilities):

- High-confidence band ($p_{max} \ge \tau_{high}$): use pseudo-label as a positive training sample.
- Medium-confidence band ($\tau_{mid} \le p_{max} < \tau_{high}$): do not trust pseudo-label strongly; apply NegL (entropy-gated) and/or downweight.
- Low-confidence band ($p_{max} < \tau_{mid}$): ignore for training (monitor only).

Recommended default starting values (to be tuned by screening):

- $\tau_{high}=0.90$
- $\tau_{mid}=0.50$

### 4.4 Stability filters (critical for real-time FER)

To avoid training on flicker/transition frames:

- Apply a “stable frame” filter during buffer building (e.g., require the predicted label to be stable for N frames, or match the smoothed label).
- Use `--min-frame-gap` and per-class caps to reduce correlation/overfitting.

### 4.5 Safety and promotion rule (must-have)

This repo’s rule is: **no promotion without gates**.

- Gate A (deployment-facing): replay-based scoring on the same labeled webcam session (`demo/outputs/*/score_results.json`).
- Gate B (offline regression): eval-only test on `Training_data_cleaned/classification_manifest_eval_only.csv` (`outputs/evals/students/*/reliabilitymetrics.json`).
- If any gate fails: rollback to the base checkpoint.

---

## 5) Status and next actions

### 5.1 Status

- The pipeline is working end-to-end (record → score → buffer → fine-tune → offline gate).
- Both attempted adaptation policies (head-only, BN-only at the tested settings) fail the offline eval-only safety gate.
- Webcam gate for BN-only is still pending (we have not yet recorded/scored a labeled run using the BN-only checkpoint).
- Self-learning with **pseudo-labels** is planned; current buffer work is primarily manual-labeled and/or conservative fine-tuning.
- NegL is implemented and can be enabled via flags. A first manifest-driven NegL + self-learning attempt was executed on 2026-02-21 (see addendum above / final report Section 4.9.2), but it did not yet yield a deployment-facing improvement.

### 5.2 Next actions (highest priority)

1) Run a labeled webcam session using `outputs/students/FT_webcam_bn_20260126_1/best.pt` and score it.
2) If it improves webcam but still fails offline:
   - reduce update size (e.g., lower LR to 5e-6, keep 1 epoch)
   - tighten buffer sampling (stable-only, bigger min-frame-gap, smaller per-class cap)
3) Only after the safe loop is stable: implement NegL for medium-confidence frames (off by default) and run a small ablation.

4) Convert “CE feels most stable” into a repeatable artifact-backed A/B:
   - record one labeled session once
   - replay/score CE vs KD+LP vs DKD with fixed demo parameters and a comparable temperature policy

---

## 6) Feb 2026 update: KD baseline vs KD + LP-loss (offline gates) — Executed (evidence)

On 2026-02-05 we ran two short-budget KD screenings (5 epochs) and evaluated both checkpoints on the offline gates used by this track.

### 5.1 Checkpoints

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`

### 5.2 Offline eval-only (safety gate)

Sources:

- KD baseline: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__classification_manifest_eval_only__test__20260205_163424/reliabilitymetrics.json`
- KD + LP: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__classification_manifest_eval_only__test__20260205_171945/reliabilitymetrics.json`

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.5162321 | 0.4385411 | 0.0217606 | 1.2961859 |
| KD + LP | 0.5207738 | 0.4411229 | 0.0374865 | 1.2773255 |

### 5.3 ExpW (target-domain proxy)

Sources:

- KD baseline: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__expw_full_manifest__test__20260205_163538/reliabilitymetrics.json`
- KD + LP: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__expw_full_manifest__test__20260205_172039/reliabilitymetrics.json`

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.6311145 | 0.4595847 | 0.0276567 | 1.0635237 |
| KD + LP | 0.6356902 | 0.4583109 | 0.0197645 | 1.0421315 |

Interpretation constraint:

- This does not yet establish a “domain-shift win” for LP-loss; ExpW macro-F1 does not improve in this screening.
- These checkpoints can still be used as candidate bases for the webcam loop, but promotion still depends on the eval-only gate + same-session webcam re-scoring.
