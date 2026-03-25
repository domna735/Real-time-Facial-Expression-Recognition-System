
# Short Discussion (2026-02-06) — Domain Shift + Real-time Stability in FER

## 1) What problem are we seeing?

**EN**

- We see a classic FER **domain shift** problem: $P_{train}(x) \neq P_{webcam}(x)$. Webcam frames differ in lighting, pose, motion blur, compression, background, and (importantly) **face-crop jitter**.
- We also see a **deployment objective mismatch**:
	- Offline evaluation optimizes macro-F1 / per-class F1 on curated splits.
	- Real-time UX optimizes “looks correct + doesn’t flicker” under noisy streaming input.
	- Therefore different checkpoints can rank differently offline vs live.

**ZH (繁體)**

- 目前觀察到的是典型的 FER **Domain Shift**：$P_{train}(x) \neq P_{webcam}(x)$。Webcam 影像在光線、姿態、運動模糊、壓縮、背景，以及（很關鍵的）**人臉框抖動/裁切抖動**方面都與訓練資料不同。
- 同時存在 **部署目標不一致（objective mismatch）**：
	- 離線評估主要最佳化 macro-F1 / per-class F1。
	- 即時使用體驗更重視「看起來對 + 不閃爍」且要能承受串流噪聲。
	- 所以同一批 checkpoint 在離線與即時的排名可能不同。

---

## 2) Evidence we already have (artifact-grounded)

**EN**

- Offline gates and domain-shift proxies exist and are reproducible:
	- **Eval-only gate**: `Training_data_cleaned/classification_manifest_eval_only.csv`
	- **ExpW proxy**: `Training_data_cleaned/expw_full_manifest.csv`
	- Outputs are stored as `outputs/evals/students/*/reliabilitymetrics.json`.
- Feb-2026 KD vs KD+LP (5-epoch screening) produced post-eval artifacts on eval-only + ExpW (see `research/process_log/Feb process log/Feb_week1_process_log.md`).
- Feb-06 real-time demo subjective observation:
	- **CE** feels more stable (less flicker) and more accurate than KD+LP and DKD in live webcam use.
	- The deployment-facing note is recorded in `research/Real time demo/real time demo report.md`.

**ZH (繁體)**

- 我們已有可重現的離線 gate 與 domain-shift proxy：
	- **Eval-only gate**：`Training_data_cleaned/classification_manifest_eval_only.csv`
	- **ExpW proxy**：`Training_data_cleaned/expw_full_manifest.csv`
	- 結果以 `outputs/evals/students/*/reliabilitymetrics.json` 保存。
- Feb-2026 KD vs KD+LP（5 epoch screening）已經有 post-eval 產生 eval-only + ExpW 的 artifact（見 `research/process_log/Feb process log/Feb_week1_process_log.md`）。
- Feb-06 即時 demo 的主觀觀察：
	- **CE** 在 webcam 使用時看起來更穩（較少閃爍）且更像正確，相較 KD+LP 與 DKD。
	- 已在 `research/Real time demo/real time demo report.md` 記錄。

---

## 3) Why CE can feel better in real time (practical hypotheses)

**EN**

- Real-time stability depends on **probability margins**, not just argmax.
- Our demo uses EMA + hysteresis (+ voting) on probabilities (`demo/realtime_demo.py`). Small differences in top-2 margins can cause big flicker changes.
- Temperature scaling (`logits / T`) changes probability sharpness (argmax unchanged) and therefore changes EMA/hysteresis dynamics.
- KD/DKD may inherit teacher uncertainty on webcam-like frames, producing more near-ties under crop jitter.

**ZH (繁體)**

- 即時穩定性主要取決於 **機率邊際（top-1 vs top-2 margin）**，不只是 argmax。
- Demo 透過 EMA + hysteresis（+ voting）處理機率（`demo/realtime_demo.py`）。top-2 margin 的小差異就可能造成明顯閃爍差異。
- Temperature scaling（`logits / T`）會改變機率尖銳度（argmax 不變），因此會影響 EMA/hysteresis 的「黏住/切換」行為。
- KD/DKD 可能在 webcam 類型影像上繼承 teacher 不確定性，在 crop jitter 下容易產生 near-tie。

---

## 4) Paper-informed methods we can try (mapped to our repo constraints)

### 4.1 TENT (test-time entropy minimization)

**EN**

- Idea: adapt on target stream by minimizing prediction entropy, updating only norm affine params $(\gamma,\beta)$.
- Risks: confirmation bias / collapse, small batch instability.
- Our safe mapping: micro-batch updates + frequent eval-only gate + rollback.

**ZH (繁體)**

- 核心：在 target 串流上做 entropy minimization，只更新 norm affine 參數 $(\gamma,\beta)$。
- 風險：confirmation bias / collapse，小 batch 會不穩。
- 安全落地：用 micro-batch 更新 + 常做 eval-only gate + 不通過就 rollback。

### 4.2 “Stable TTA” (SAR-like: reliable filter + recovery)

**EN**

- Add two safety rails on top of TENT:
	- **Reliable sample selection**: only adapt on low-entropy frames (skip noisy frames).
	- **Recovery/reset**: detect collapse and reset to the base checkpoint.
- This aligns with our existing “gate + rollback” philosophy.

**ZH (繁體)**

- 在 TENT 之上增加兩個穩定機制：
	- **可靠樣本篩選**：只用低 entropy 的 frame 來更新，跳過不確定樣本。
	- **Recovery/reset**：偵測 collapse 後回復 base checkpoint。
- 與我們的「gate + rollback」策略高度一致。

### 4.3 DANN (domain-adversarial training)

**EN**

- Train a domain-invariant representation using a Gradient Reversal Layer (GRL).
- Minimal screening config in this repo:
	- Source (labeled): HQ-train
	- Target (unlabeled): ExpW train split (keep ExpW test held out)
	- Evaluate: ExpW test + eval-only gate
- Risk: too-large domain weight $\lambda$ harms label discrimination.

**ZH (繁體)**

- 透過 GRL 做 domain adversarial training，學到 domain-invariant 表徵。
- Repo 內最小 screening：
	- Source（有標籤）：HQ-train
	- Target（無標籤）：ExpW train（ExpW test 保持不參與訓練）
	- 評估：ExpW test + eval-only gate
- 風險：$\lambda$ 過大會犧牲分類能力。

### 4.4 Domain Density Transformations (invariance under transforms)

**EN**

- Full paper approach uses learned domain transforms (e.g., GAN/StarGAN) which is high-cost.
- Low-cost proxy we can try first: representation consistency under augmentations / “domain-mix” transforms.

**ZH (繁體)**

- 原論文較完整做法需要學 domain transform（GAN/StarGAN），成本高。
- 可先做低成本 proxy：用 augmentation / domain-mix transform 做 representation consistency loss。

### 4.5 Reliable Crowdsourcing + Locality-Preserving (LP) learning

**EN**

- We already implemented LP-loss (paper #5 Track A) and ran KD vs KD+LP screening.
- Current evidence: LP-loss mainly improved calibration signals (TS ECE/NLL) but did not improve ExpW raw macro-F1 under the tested short-budget config.
- Next: tune weight (e.g., 0.001) / change embedding (logits vs penultimate) / improve neighbor selection stability.

**ZH (繁體)**

- 我們已實作 LP-loss（paper #5 Track A）並做 KD vs KD+LP screening。
- 目前證據顯示：LP-loss 在校準（TS ECE/NLL）上有正向訊號，但在目前短 budget 設定下，ExpW raw macro-F1 沒有提升。
- 下一步：調小 weight（如 0.001）/ 改 embedding（logits vs penultimate）/ 改善 neighbor 選取穩定性。

---

## 5) Proposed next experiments (strictly evidence-first + gate-safe)

**EN**

1) **Replay-based real-time A/B** (make the subjective result repeatable)
	 - Record one labeled webcam session once.
	 - Replay/score CE vs KD+LP vs DKD with fixed parameters and a comparable temperature rule.
	 - Artifacts: `demo/outputs/*/score_results.json` + `per_frame.csv`.
2) **SAR-lite TTA screening** (safe TENT)
	 - Adapt only BN affine params on webcam buffer or ExpW streaming batches.
	 - Reliable filter + recovery + eval-only gate after each chunk.
3) **DANN screening**
	 - Small $\lambda$ ramp, short budget (3 epochs), evaluate ExpW + eval-only.

**ZH (繁體)**

1) **Replay-based real-time A/B**（把主觀結果變成可重現）
	 - 只錄一次 labeled webcam session。
	 - 固定參數與溫度規則，重播比較 CE vs KD+LP vs DKD。
	 - 產出 artifact：`demo/outputs/*/score_results.json` + `per_frame.csv`。
2) **SAR-lite TTA screening**（安全版 TENT）
	 - 只更新 BN affine，在 webcam buffer 或 ExpW streaming batches 上做。
	 - 加可靠樣本篩選 + recovery，並且每個 chunk 都跑 eval-only gate。
3) **DANN screening**
	 - 小 $\lambda$ ramp、短 budget（3 epochs），評估 ExpW + eval-only。

---

## 5.5 Main track: Self-Learning + Negative Learning (NegL) for webcam domain shift

**EN**

- Self-Learning: use high-confidence pseudo-label frames from webcam buffer to fine-tune the student toward the target domain.
- NegL: apply complementary-label negative learning on uncertain (high-entropy / medium-confidence) frames to reduce confirmation bias.
- Where it is implemented:
	- Buffer building: `scripts/build_webcam_buffer.py` from `demo/outputs/<run>/per_frame.csv`
	- NegL loss: `src/fer/negl/losses.py`
	- Training flags: `scripts/train_student.py --use-negl --negl-gate entropy ...`
- Promotion rule: pass BOTH replay-based webcam scoring and offline eval-only gate; otherwise rollback.

**ZH (繁體)**

- Self-Learning：用 webcam buffer 入面高信心 pseudo-label frame 做 fine-tune，令模型更貼近 target domain。
- NegL：對不確定（高 entropy / 中信心）frame 用 complementary-label negative learning，減少 confirmation bias。
- Repo 對應位置：
	- Buffer：`scripts/build_webcam_buffer.py`（讀 `demo/outputs/<run>/per_frame.csv`）
	- NegL loss：`src/fer/negl/losses.py`
	- Training 參數：`scripts/train_student.py --use-negl --negl-gate entropy ...`
- 推進規則：必須同時過「重播 webcam 評分」同「eval-only gate」，否則 rollback。

---

## 5.6 Papers: what we can use vs what challenges us

**EN**

- Use:
	- TENT/SAR: reliable sample selection + recovery/reset supports our gating philosophy.
	- DANN: alternative if pseudo-label noise is too high.
	- Density transforms: suggests invariance regularization (proxy via domain-mix augmentations).
- Challenge:
	- TENT/SAR show online adaptation can collapse under small batches/non-stationary streams.
	- BN can be unstable; our BN-only attempt already shows offline regression risk.

**ZH (繁體)**

- 可用：
	- TENT/SAR：可靠樣本挑選 + recovery/reset，支持我哋 gate/rollback。
	- DANN：當 pseudo-label 太嘈時嘅替代方案。
	- Density transforms：不變性 regularization（可先用 domain-mix augmentation proxy）。
- 挑戰：
	- TENT/SAR 指出小 batch/非平穩會 collapse。
	- BN 可能不穩；我哋 BN-only 已見 offline regression 風險。

---

## 6) What we want to tell the supervisor (one-line takeaway)

**EN**

- Offline macro-F1 and real-time stability can disagree under domain shift; we will keep CE as demo default for stability, and we will convert the preference into reproducible artifacts via one replay-based labeled session, while testing safe TTA (SAR-lite) and DANN under strict eval-only gating.

**ZH (繁體)**

- 在 domain shift 下，離線 macro-F1 與即時穩定性可能不一致；短期先用 CE 當 demo default（穩定性 KPI），並用一次 replay-based labeled session 把主觀偏好變成可重現 artifact，同時在嚴格 eval-only gate 下測試安全版 TTA（SAR-lite）與 DANN。

