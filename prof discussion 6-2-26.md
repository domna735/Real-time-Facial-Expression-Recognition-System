## Discussion (2026-02-06) — Domain Shift + Real-time Stability in FER

## 0) 60-second summary (what we want to align on)

**EN**

- We have a FER **domain shift**: $P_{train}(x) \neq P_{webcam}(x)$, with major noise sources (lighting/blur/compression) and especially **face-crop jitter**.
- We observed **objective mismatch**: offline macro-F1 does not fully capture real-time “looks correct + doesn’t flicker”.
- Short-term: keep **CE checkpoint as demo default** (best perceived stability so far).
- Next: convert the subjective preference into **reproducible artifacts** via a single replay-based labeled session, and only promote any adaptation if it passes both **Gate A (replay stability/accuracy)** and **Gate B (eval-only regression)**.

**ZH (繁體)**

- 目前是典型 FER **Domain Shift**：$P_{train}(x) \neq P_{webcam}(x)$，主要差異來自光線/模糊/壓縮等噪聲，尤其是 **人臉裁切抖動（face-crop jitter）**。
- 我們遇到 **部署目標不一致**：離線 macro-F1 無法完整反映即時「看起來對 + 不閃爍」。
- 短期策略：demo 先以 **CE checkpoint** 作為預設（目前主觀體感最穩）。
- 下一步：用一次 replay-based labeled session 產生 **可重現 artifacts**；所有自適應/微調都必須同時通過 **Gate A（replay 指標）** 與 **Gate B（eval-only 回歸）** 才能提升版本。

---

## 0.1 Meeting goals (decisions we want from supervisor)

**EN**

1) Confirm our **primary deployment KPI** (stability vs accuracy trade-off) and whether replay-based scoring is acceptable as an official gate.
2) Approve the **main track**: Self-Learning + NegL with strict filtering + rollback.
3) Choose the **secondary screening track** to prioritize after replay A/B: SAR-lite (safe TENT) vs DANN (domain adversarial) vs LP-loss tuning.

**ZH (繁體)**

1) 確認部署端的 **主要 KPI**（穩定性 vs 準確度取捨），以及 replay-based scoring 是否可作為正式 gate。
2) 同意主線：在嚴格篩選 + rollback 下做 **Self-Learning + NegL**。
3) 決定 replay A/B 後要優先做哪條次線 screening：**SAR-lite**（安全版 TENT）/ **DANN**（domain adversarial）/ **LP-loss 調參**。

---

## 1) What problem are we seeing?

**EN**

- Domain shift: webcam frames differ in lighting/pose/motion blur/compression/background and especially face-crop jitter.
- Objective mismatch: offline macro-F1/per-class F1 vs real-time “looks correct + doesn’t flicker”.

**ZH (繁體)**

- Domain shift：webcam 影像在光線、姿態、動態模糊、壓縮、背景，以及 **人臉裁切抖動** 上，與訓練資料分佈不同。
- Objective mismatch：離線 macro-F1/per-class F1 與即時「看起來對 + 不閃爍」指標不一致。

---

## 2) Evidence we already have (artifact-grounded)

**EN**

- Offline reproducible gates / proxies:
	- Eval-only gate: `Training_data_cleaned/classification_manifest_eval_only.csv`
	- ExpW proxy: `Training_data_cleaned/expw_full_manifest.csv`
	- Metrics artifacts: `outputs/evals/students/*/reliabilitymetrics.json`
- Feb-2026 KD vs KD+LP screening: post-eval artifacts exist (see `research/process_log/Feb process log/Feb_week1_process_log.md`).
- Feb-06 real-time subjective observation: CE feels more stable/accurate than KD+LP and DKD (recorded in `research/Real time demo/real time demo report.md`).

**ZH (繁體)**

- 已具備可重現的離線 gate / domain-shift proxy：
	- Eval-only gate：`Training_data_cleaned/classification_manifest_eval_only.csv`
	- ExpW proxy：`Training_data_cleaned/expw_full_manifest.csv`
	- 指標 artifacts：`outputs/evals/students/*/reliabilitymetrics.json`
- Feb-2026 KD vs KD+LP（短 budget screening）已產生 post-eval artifacts（見 `research/process_log/Feb process log/Feb_week1_process_log.md`）。
- Feb-06 即時 demo 主觀觀察：CE 較 KD+LP、DKD 更穩/更像正確（已記錄於 `research/Real time demo/real time demo report.md`）。

---

## 3) Why CE can feel better in real time (practical hypotheses)

**EN**

- Real-time stability depends on **probability margins** (top-1 vs top-2), not just argmax.
- Our demo applies EMA + hysteresis (+ optional voting) over probabilities (`demo/realtime_demo.py`), so small margin changes can produce big flicker differences.
- Temperature scaling (`logits / T`) changes probability sharpness (argmax unchanged) and therefore changes EMA/hysteresis dynamics.
- KD/DKD can inherit teacher uncertainty on webcam-like frames → more near-ties under crop jitter.

**ZH (繁體)**

- 即時穩定性取決於 **機率邊際（top-1 vs top-2）**，不只是 argmax。
- Demo 對機率做 EMA + hysteresis（+ 可選 voting）（`demo/realtime_demo.py`），因此邊際的微小差異可能造成明顯閃爍差異。
- Temperature scaling（`logits / T`）會改變機率尖銳度（argmax 不變），進而影響 EMA/hysteresis 的「黏住/切換」行為。
- KD/DKD 在 webcam 類型影像上可能繼承 teacher 的不確定性，在裁切抖動下更容易 near-tie。

---

## 4) Proposed experiments (evidence-first + gate-safe)

### 4.1 One table (what we run, what we get)

| Experiment | Purpose | Update scope | Must-pass gates | Primary artifacts |
|---|---|---|---|---|
| Replay-based real-time A/B | Make the subjective CE preference reproducible | No training; just replay/score | Gate A | `demo/outputs/*/score_results.json`, `per_frame.csv` |
| Self-Learning + NegL (main track) | Reduce webcam appearance gap safely | Fine-tune student (small LR) + NegL on medium-confidence | Gate A + Gate B | new student ckpt + eval artifacts |
| SAR-lite TTA (safe TENT) | Fast adaptation with safety rails | BN affine only + reliable filter + recovery | Gate B (and ideally Gate A) | eval artifacts per chunk |
| DANN screening | Domain-invariant features w/ unlabeled target | Add GRL head, small $\lambda$ ramp | Gate B + ExpW | ExpW + eval artifacts |
| LP-loss tuning (optional) | Stabilize embedding geometry/calibration | Adjust LP weight / embedding choice | Gate B + ExpW | ExpW + eval artifacts |

**EN**

- Priority order: (1) Replay A/B → (2) Main track (Self-Learning + NegL) → (3) SAR-lite or DANN as secondary track.

**ZH (繁體)**

- 優先順序：(1) Replay A/B → (2) 主線（Self-Learning + NegL）→ (3) 次線再選 SAR-lite 或 DANN。

---

## 5) Main track: Domain Shift Improvement via Self-Learning + Negative Learning (NegL)

This is the plan we want to align on.

### 5.1 What Self-Learning is (and why it can help)

**EN**

- Self-Learning: generate **pseudo-labels** on webcam frames, then fine-tune the student on those target-domain crops.
- Why it can help: reduces target appearance gap (camera noise / lighting / blur / crop style) and can improve real-time behavior if we train only on **clean, stable** frames.
- Repo wiring (already supported):
	- `demo/outputs/<run>/per_frame.csv`
	- `scripts/build_webcam_buffer.py`
	- `scripts/train_student.py`

**ZH (繁體)**

- Self-Learning：先對 webcam frames 產生 **pseudo-label**，再用 target-domain crops 做 fine-tune。
- 為什麼可能有效：直接縮小 webcam 的外觀差距（鏡頭噪聲/光線/模糊/裁切風格）。關鍵是只用 **乾淨 + 穩定** frames 才能降低噪聲與偏差。
- Repo 內的銜接點（已支援）：
	- `demo/outputs/<run>/per_frame.csv`
	- `scripts/build_webcam_buffer.py`
	- `scripts/train_student.py`

### 5.2 What NegL is (and where we use it)

**EN**

- NegL here = **complementary-label negative learning**:
	- Code: `src/fer/negl/losses.py` (`complementary_negative_loss`)
	- Training flags: `--use-negl`, `--negl-weight`, `--negl-ratio`, entropy gate (`--negl-gate entropy --negl-entropy-thresh ...`).
- Intended usage (confidence-banded policy):
	- High-confidence: pseudo-label CE.
	- Medium-confidence: entropy-gated NegL (discourage wrong classes without fully trusting a hard pseudo-label).
	- Low-confidence: skip (no update).

**ZH (繁體)**

- 本 repo 的 NegL = **complementary-label negative learning**：
	- 程式：`src/fer/negl/losses.py`（`complementary_negative_loss`）
	- 參數：`--use-negl`, `--negl-weight`, `--negl-ratio`，以及 entropy gate（`--negl-gate entropy --negl-entropy-thresh ...`）。
- 預期用法（依信心分段）：
	- 高信心：用 pseudo-label 做 CE。
	- 中信心：用 entropy gate 的 NegL（在不完全相信 hard pseudo-label 的情況下，先把明顯錯的類別推開）。
	- 低信心：跳過不更新。

### 5.3 Safety gates (non-negotiable)

**EN**

- Gate A (deployment-facing): replay-based scoring on the same labeled session → `demo/outputs/*/score_results.json`.
- Gate B (offline regression): eval-only test → `outputs/evals/students/*/reliabilitymetrics.json`.
- Promote only if BOTH gates pass; otherwise rollback to base checkpoint.

**ZH (繁體)**

- Gate A（部署指標）：同一段 labeled session 的重播評分 → `demo/outputs/*/score_results.json`。
- Gate B（離線回歸）：eval-only 測試 → `outputs/evals/students/*/reliabilitymetrics.json`。
- 兩個 gate 都通過才提升版本；否則 rollback 到 base checkpoint。

---

## 6) What we can use from papers (and what challenges our plan)

### 6.1 Use (supports our plan)

**EN**

- TENT: supports entropy/confidence signals and restricting updates to small parameter subsets.
- SAR (stable TTA): supports reliable sample selection + recovery/reset (fits our gate/rollback philosophy).
- DANN: alternative path when pseudo-labels are too noisy.
- Domain density transforms: motivates invariance under transforms; low-cost proxy = consistency under domain-mix augmentations.
- Crowdsourcing + DLP: supports uncertainty-aware training (don’t trust every frame as a hard label).

**ZH (繁體)**

- TENT：支持以 entropy/confidence 作為 adaptation signal，並限制更新參數集合。
- SAR（stable TTA）：支持加入可靠樣本挑選 + recovery/reset（與 gate/rollback 一致）。
- DANN：當 pseudo-label 太嘈時的替代路線。
- Domain density transforms：提出 transform invariance；低成本 proxy 可用 domain-mix augmentation consistency。
- Crowdsourcing + DLP：指出 FER 標籤有歧義/多模態，應採用 uncertainty-aware 訓練而非盲信每一幀。

### 6.2 Challenge (risks we must address)

**EN**

- Naive TTA can collapse under non-stationary streams + small batches → we must use stable-frame filters, reliable selection, and recovery.
- BN sensitivity is a known stability bottleneck → BN-only attempts can regress offline metrics.
- Pseudo-label confirmation bias → NegL helps only if gating/filtering is correct.

**ZH (繁體)**

- 在非平穩串流 + 小 batch 下，naive TTA 可能 collapse → 必須要穩定 frame 篩選、可靠樣本挑選與 recovery。
- BN 對 wild online setting 很敏感 → BN-only 自適應可能造成離線指標回歸。
- Pseudo-label confirmation bias → NegL 只能在 gate/filter 設計正確時才是保護。

---

## 7) Questions to ask the supervisor (discussion checklist)

**EN**

1) Should we officially treat “real-time stability” as a first-class KPI alongside macro-F1?
2) Is a single replay-based labeled session acceptable as Gate A, if we keep parameters fixed and store artifacts?
3) For the main track, what confidence bands (high/medium/low) and what rollback criteria do you consider acceptable?
4) Which secondary track should we prioritize after replay A/B: SAR-lite vs DANN vs LP-loss tuning?

**ZH (繁體)**

1) 是否同意把「即時穩定性」與 macro-F1 一樣作為主要 KPI？
2) 若固定參數並保存 artifacts，「一次 replay-based labeled session」是否可作為 Gate A？
3) 主線的信心分段（高/中/低）與 rollback 條件，您認為應該如何設定才安全？
4) Replay A/B 後次線優先做哪個：SAR-lite / DANN / LP-loss 調參？

