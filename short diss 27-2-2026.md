# Short Discussion Note (27-02-2026) — FER Project Progress + Negative Results + Next Steps

（繁體中文）簡短討論備忘（27-02-2026）— FER 專案進度、負面結果與下一步

Audience: Prof. Lam  
（繁體中文）對象：Lam 教授  
Date: 2026-02-27  
（繁體中文）日期：2026-02-27  
Repo: Real-time Facial Expression Recognition System v2 (restart)
（繁體中文）專案：即時臉部表情辨識系統 v2（重啟版）

---

## 1) What I want to discuss in this meeting

（繁體中文）1）本次會議我想討論的重點

1. **Where the project is now (process + system status)**: what is already rebuilt and reproducible.
   （繁體中文）**目前專案進度（流程 + 系統狀態）**：哪些部分已經重建完成並且可重現。
2. **Key negative results (detailed) + reasons**: especially the webcam domain-shift loop that *passed offline gate* but *regressed on deployment-facing replay*.
   （繁體中文）**關鍵負面結果（詳細）與原因**：特別是網路攝影機（webcam）領域轉移迴圈 *通過離線 gate*，但在 *部署導向的同一段錄影重播評分* 上反而退步。
3. **Evidence-backed comparison results**: teacher hard-gate robustness + student (KD/DKD) objective comparisons + the webcam A/B numbers.
   （繁體中文）**可追溯證據的比較結果**：Teacher 的 hard-gate 強健性 + Student（KD/DKD）目標函數比較 + webcam A/B 的數字。
4. **Next-step plan**: the smallest set of controlled experiments (“one knob at a time”) that can convert negative findings into publishable, defensible conclusions.
   （繁體中文）**下一步計畫**：用最小、最可控的實驗集合（一次只調一個旋鈕）把負面結果轉成可發表、可辯護的結論。

---

## 2) Current process (how we work now)

（繁體中文）2）目前的工作流程（我們現在怎麼做）

### 2.1 Evidence-first workflow (the main engineering change)

（繁體中文）2.1 證據優先（evidence-first）的工作流程（主要工程改變）

All claims are tied to on-disk artifacts:

（繁體中文）所有說法都必須對應到磁碟上的實驗產物（artifact）：

- **Training runs** live under `outputs/students/**` or `outputs/teachers/**`.
  （繁體中文）**訓練 run** 會存放在 `outputs/students/**` 或 `outputs/teachers/**`。
- **Evaluations** are stored as `outputs/evals/**/reliabilitymetrics.json` (reproducible, auditable).
  （繁體中文）**評估結果** 會存為 `outputs/evals/**/reliabilitymetrics.json`（可重現、可稽核）。
- **Real-time demo** sessions are logged under `demo/outputs/<session_id>/`:
  （繁體中文）**即時 demo** 的每次 session 會記錄在 `demo/outputs/<session_id>/`：
  - `per_frame.csv` (manual labels + predictions)
    （繁體中文）`per_frame.csv`（人工標註 + 預測）
  - `score_results.json` (raw/smoothed metrics + stability/jitter)
    （繁體中文）`score_results.json`（raw / smoothed 指標 + 穩定度/抖動 jitter）

This is important because many “wins/losses” in domain adaptation can be caused by hidden confounds (preprocessing mismatch, BatchNorm drift).

（繁體中文）這非常重要，因為領域適應（domain adaptation）中的「變好/變差」常常其實是隱藏干擾因素造成（例如 preprocessing 不一致、BatchNorm 漂移）。

### 2.2 What has already been rebuilt / implemented

（繁體中文）2.2 已重建 / 已實作的部分

1. **Teacher → Student pipeline** (CE / KD / DKD students) + evaluation + compare tables.
   （繁體中文）**Teacher → Student 流程**（CE / KD / DKD student）+ 評估 + compare tables。
2. **Cross-dataset gates**:
   （繁體中文）**跨資料集 gate（守門）**：
   - Offline safety gate: `Training_data_cleaned/classification_manifest_eval_only.csv`
     （繁體中文）離線安全 gate：`Training_data_cleaned/classification_manifest_eval_only.csv`
   - Cross-dataset proxy: `Training_data_cleaned/expw_full_manifest.csv`
     （繁體中文）跨資料集（in-the-wild）代理測試：`Training_data_cleaned/expw_full_manifest.csv`
3. **Webcam scoring protocol** (deployment-facing):
   （繁體中文）**Webcam 評分協議**（部署導向）：
   - fixed recorded session replay
     （繁體中文）固定使用同一段錄影 session 重播
   - metrics reported for both raw and smoothed predictions
     （繁體中文）同時回報 raw 與 smoothed 預測的指標
   - jitter flips/min tracked
     （繁體中文）追蹤 jitter flips/min（每分鐘翻轉次數）
4. **Domain-shift extension**: a safety-gated webcam adaptation loop:
   （繁體中文）**領域轉移延伸**：具安全 gate 的 webcam 適應迴圈：
   - build a self-learning buffer manifest from a recorded session
     （繁體中文）從錄影 session 建立 self-learning buffer manifest
   - conservative fine-tune candidate
     （繁體中文）保守式 fine-tune 候選模型
   - require (A) offline non-regression gate AND (B) same-session replay improvement before promoting
     （繁體中文）升級候選模型前必須同時滿足（A）離線不退步 gate（non-regression）與（B）同一段 session 重播評分有提升

Key process log entries:

- `research/process_log/Feb process log/Feb_week3_process_log.md`

（繁體中文）關鍵流程紀錄（process log）：

- `research/process_log/Feb process log/Feb_week3_process_log.md`

Main report (full narrative):

- `research/final report/final report version 2.md`

（繁體中文）主報告（完整敘事）：

- `research/final report/final report version 2.md`

Dedicated negative-result deliverable:

- `research/final report/negative result report/negative result report.md`

（繁體中文）獨立的負面結果報告（deliverable）：

- `research/final report/negative result report/negative result report.md`

### 2.3 Paper comparison progress (what exists now)

（繁體中文）2.3 論文比較進度（目前已具備的東西）

Status today:

（繁體中文）目前狀態：

- A **fair internal comparison protocol** is written and defines what “apples-to-apples” means inside this repo:
- A **fair internal comparison protocol** is written and defines what “apples-to-apples” means inside this repo: `research/fair_compare_protocol.md`
   （繁體中文）已撰寫 **公平內部比較協議**，定義在本 repo 內什麼叫做「蘋果對蘋果」：`research/fair_compare_protocol.md`
- Paper PDFs have been converted into searchable on-disk text artifacts (so claims can be quoted and audited): `outputs/paper_extract/`
   （繁體中文）已把論文 PDF 轉成可搜尋的文字產物（方便引用與稽核）：`outputs/paper_extract/`

What is *not yet complete* (so we do not over-claim):

（繁體中文）尚未完成的部分（所以我們不會過度宣稱）：

- A full, protocol-matched reproduction for each paper is not finished.
   （繁體中文）每篇論文的「完全 protocol-match 重現」尚未完成。
- The comparison matrix is still being finalized into a single supervisor-facing table (dataset/splits/metrics/TTA/etc.).
   （繁體中文）比較矩陣仍在整理成一張面向導師的總表（資料集/切分/指標/TTA 等）。

What this enables next:

（繁體中文）因此下一步可做：

- We can now pick 1 anchor dataset (e.g., RAF-DB basic) and run the exact same evaluation pipeline across baseline methods and paper-inspired knobs, producing consistent compare tables.
   （繁體中文）我們可以先選 1 個 anchor dataset（例如 RAF-DB basic），用完全相同的評估流程跑 baseline 方法與論文啟發的方法旋鈕，產生一致的 compare tables。

---

## 3) Core negative result (NR-A1): webcam self-learning + NegL passed gate but regressed on replay

（繁體中文）3）核心負面結果（NR-A1）：webcam 自學習 + NegL 通過 gate 但在重播評分退步

This is currently the **most important negative result**, because it directly affects deployment-facing claims.

（繁體中文）這是目前 **最重要的負面結果**，因為它直接影響部署導向的結論。

### 3.1 What was the goal

（繁體中文）3.1 目標是什麼

Reduce **webcam domain shift** using a conservative adaptation loop:

（繁體中文）使用保守式適應迴圈來降低 **webcam 領域轉移**：

- collect one labeled recorded session
  （繁體中文）收集一段已人工標註的錄影 session
- build a self-learning buffer (pseudo-label fine-tune)
  （繁體中文）建立 self-learning buffer（pseudo-label 微調）
- apply weighted CE on high-confidence pseudo-label positives
  （繁體中文）對高信心 pseudo-label 正樣本用 weighted CE
- apply manifest-driven NegL on medium-confidence samples
  （繁體中文）對中等信心樣本用 manifest 驅動的 NegL
- accept a candidate **only if** it passes both:
  （繁體中文）只有在同時通過以下兩項時才接受候選模型：
  1) offline safety gate, and
    （繁體中文）離線安全 gate
  2) identical-session replay improvement
    （繁體中文）同一段 session 重播評分有改善

### 3.2 Protocol (what makes it a fair A/B)

（繁體中文）3.2 協議（為什麼是公平的 A/B）

Baseline checkpoint:

- `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`

（繁體中文）Baseline checkpoint：

- `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`

Recorded labeled session (deployment-like objective):

- `demo/outputs/20260126_205446/`
  - baseline score: `demo/outputs/20260126_205446/score_results.json`

（繁體中文）已標註的錄影 session（部署導向目標）：

- `demo/outputs/20260126_205446/`（baseline score：`demo/outputs/20260126_205446/score_results.json`）

Self-learning buffer manifest built from the same session:

- `demo/outputs/20260126_205446/buffer_selflearn/manifest.csv`

（繁體中文）由同一段 session 建立的 self-learning buffer manifest：

- `demo/outputs/20260126_205446/buffer_selflearn/manifest.csv`

Offline safety gate artifacts (eval-only manifest):

- baseline gate dir: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/baseline/`
- adapted gate dir: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/adapted_clahe_head_frozebn/`

（繁體中文）離線安全 gate 產物（eval-only manifest）：

- baseline gate dir：`outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/baseline/`
- adapted gate dir：`outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/adapted_clahe_head_frozebn/`

Adapted checkpoint:

- `outputs/students/DA/mnv3_webcamselflearn_negl_clahe_head_frozebn_20260221_211025/best.pt`

（繁體中文）Adapted checkpoint：

- `outputs/students/DA/mnv3_webcamselflearn_negl_clahe_head_frozebn_20260221_211025/best.pt`

Deployment-facing A/B replay score (same labels, same session):

- adapted score: `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`

（繁體中文）部署導向 A/B 重播評分（同標註、同 session）：

- adapted score：`demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`

Fairness controls (why this is not a “measurement artifact” result):

（繁體中文）公平性控制（為什麼這不是「量測假象」）：

- **Preprocessing matched**: adaptation uses consistent CLAHE behavior (`use_clahe=True`).
  （繁體中文）**Preprocessing 一致**：adaptation 與 baseline 使用一致的 CLAHE 行為（`use_clahe=True`）。
- **BatchNorm drift controlled**: BN running stats frozen during conservative tuning.
  （繁體中文）**BatchNorm 漂移已控制**：保守式 tuning 時凍結 BN running stats。
- **Same manual labels reused** in replay, not re-labeled.
  （繁體中文）重播時 **沿用同一份人工標註**，不重新標註。

### 3.3 What happened (evidence-backed numbers)

（繁體中文）3.3 發生了什麼（可追溯證據的數字）

Smoothed metrics (same `scored_frames=4154`):

（繁體中文）Smoothed 指標（同 `scored_frames=4154`）：

- Baseline: accuracy **0.5879**, macro-F1 **0.5248**, minority-F1(lowest-3) **0.1609**, jitter **14.86** flips/min
- Adapted:  accuracy **0.5269**, macro-F1 **0.4667**, minority-F1(lowest-3) **0.1384**, jitter **14.16** flips/min

（繁體中文）

- Baseline：accuracy **0.5879**、macro-F1 **0.5248**、minority-F1（最低 3 類）**0.1609**、jitter **14.86** flips/min
- Adapted：accuracy **0.5269**、macro-F1 **0.4667**、minority-F1（最低 3 類）**0.1384**、jitter **14.16** flips/min

Interpretation:

（繁體中文）解讀：

- The adapted model is **not beneficial** for the labeled webcam replay objective.
- The offline gate being passed **is necessary but not sufficient** for deployment-like improvement.

（繁體中文）

- Adapted 模型對已標註的 webcam 重播目標 **沒有帶來改善**。
- 通過離線 gate **是必要條件但不是充分條件**，不能保證部署導向的改善。

### 3.4 Why it likely happened (hypotheses to test)

（繁體中文）3.4 可能原因（需要用實驗驗證的假設）

These are plausible mechanisms consistent with the pipeline and the buffer policy.

（繁體中文）以下是與目前 pipeline 與 buffer policy 一致、且合理的機制假設（需透過控制實驗確認）。

1. **Small, correlated buffer**
   - Single-session adaptation can overfit to subject/lighting/pose.
   - Even if eval-only macro-F1 is stable, decision margins relevant to temporal smoothing can degrade.

   （繁體中文）**小且高度相關的 buffer**
   - 單一 session 的適應容易對特定人臉/光照/姿態過擬合。
   - 即使 eval-only macro-F1 看起來穩定，與時間平滑（smoothing）相關的決策邊界/機率邊際仍可能變差。

2. **Pseudo-label noise + transition frames**
   - “stable frame” heuristics reduce noise but do not eliminate ambiguity.
   - A small number of wrong pseudo-labels can shift boundaries.

   （繁體中文）**Pseudo-label 噪聲 + 轉換過渡幀**
   - 「穩定幀」規則可以降低噪聲，但無法完全消除模糊過渡狀態。
   - 少量錯誤 pseudo-label 也可能推動決策邊界偏移。

3. **NegL target-policy risk (medium-confidence frames)**
   - Current implementation uses medium-confidence samples as NegL-only with `weight=0` and `neg_label=<predicted_label>`.
   - If the prediction is *often correct-but-uncertain*, treating it as a “negative label” can push probability away from the correct class and harm macro-F1.

   （繁體中文）**NegL target-policy 的風險（中等信心幀）**
   - 目前實作把中等信心樣本做 NegL-only，並設定 `weight=0`、`neg_label=<predicted_label>`。
   - 若該 predicted label 常常是「其實正確但不夠有信心」，把它當作 negative label 可能會把機率從正確類別推開，導致 macro-F1 下降。

4. **Objective mismatch**
   - Offline gates (eval-only / ExpW) are broad static distribution checks.
   - Webcam replay depends on probability margins + smoothing/hysteresis, so a model can pass offline non-regression but still worsen deployment metrics.

   （繁體中文）**目標不一致（objective mismatch）**
   - 離線 gate（eval-only / ExpW）是較廣泛、靜態分佈的檢查。
   - Webcam 重播評分依賴機率邊際 + smoothing/hysteresis；因此模型可能通過離線不退步，但仍讓部署指標退步。

### 3.5 Engineering lessons (already acted on)

（繁體中文）3.5 工程教訓（已採取行動修正）

Two confounds were discovered and fixed during this work:

（繁體中文）這次工作中辨識並修正了兩個關鍵干擾因素：

- **Preprocessing mismatch (CLAHE)** can invalidate A/B comparisons.
- **BatchNorm running-stat drift** can happen even in “head-only tuning” unless BN stats are frozen.

（繁體中文）

- **Preprocessing 不一致（CLAHE）** 會讓 A/B 比較失去公平性。
- **BatchNorm running-stat 漂移** 即使在「head-only tuning」也可能發生，除非凍結 BN stats。

These lessons now appear in the final report and negative result report, and the training code path was adjusted accordingly.

（繁體中文）這些教訓已寫入主報告與負面結果報告，訓練程式碼也已相應調整。

---

## 4) Negative compare results (offline): NegL / NL(proto) is not consistently beneficial under screening settings

（繁體中文）4）離線比較的負面結果：在目前的 screening 設定下，NegL / NL(proto) 並非穩定有益

This part supports the “negative results are real, not just webcam noise” claim: even offline, some auxiliary objectives regress macro-F1 or minority-F1.

（繁體中文）這段支持「負面結果是真實存在，而不只是 webcam 噪聲」的說法：即使在離線設定下，一些輔助目標也會讓 macro-F1 或 minority-F1 退步。

### 4.1 KD baseline vs KD + NegL (screening)

（繁體中文）4.1 KD baseline vs KD + NegL（screening）

Artifact table:

- `outputs/students/_compare_kd5_vs_negl5.md`

（繁體中文）對應的 compare table：

- `outputs/students/_compare_kd5_vs_negl5.md`

Summary (Raw metrics):

（繁體中文）摘要（Raw 指標）：

| Setting | Raw acc | Raw macro-F1 | Minority-F1 (lowest-3) | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| KD-only (5ep) | 0.728363 | 0.726648 | 0.697342 | 0.027051 | 0.783856 |
| KD + NegL (5ep) | 0.722364 | 0.719800 | 0.682749 | 0.039770 | 0.808534 |

Interpretation:

- Under this configuration, **NegL regresses macro-F1 and minority-F1** and also worsens TS calibration.

（繁體中文）解讀：

- 在此設定下，**NegL 會讓 macro-F1 與 minority-F1 退步**，同時 TS calibration 也變差。

### 4.2 DKD baseline vs DKD + NegL

（繁體中文）4.2 DKD baseline vs DKD + NegL

Artifact table:

- `outputs/students/_compare_dkd5_negl_vs_dkd5.md`

（繁體中文）對應的 compare table：

- `outputs/students/_compare_dkd5_negl_vs_dkd5.md`

Summary (Raw metrics):

（繁體中文）摘要（Raw 指標）：

| Setting | Raw acc | Raw macro-F1 | Minority-F1 (lowest-3) | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| DKD baseline | 0.735711 | 0.736796 | 0.704458 | 0.034764 | 0.783468 |
| DKD + NegL | 0.735060 | 0.734752 | 0.702431 | 0.034830 | 0.792553 |

Interpretation:

- Here NegL produces a **small but consistent regression** in macro-F1 and minority-F1.

（繁體中文）解讀：

- 在此對比中，NegL 造成 **小幅但一致的退步**（macro-F1 與 minority-F1）。

### 4.3 KD + (NegL + NL(proto)) example: severe collapse under one screening config

（繁體中文）4.3 KD +（NegL + NL(proto)）範例：某個 screening 設定下出現嚴重崩潰

Artifact table:

- `outputs/students/_compare_kd5_negl_nl_vs_kd5.md`

（繁體中文）對應的 compare table：

- `outputs/students/_compare_kd5_negl_nl_vs_kd5.md`

Key comparison (Raw metrics):

（繁體中文）關鍵對比（Raw 指標）：

- Reference KD-only (5ep): raw acc 0.728363, raw macro-F1 0.726648, minority-F1 0.697342
- KD + NegL + NL(proto): raw acc 0.540157, raw macro-F1 0.520387, minority-F1 0.349783

Interpretation:

- This is a strong negative result: some combined auxiliary objectives are **unstable** and can produce large regressions.
- It supports the policy “screen fast, one knob at a time, and do not over-claim.”

（繁體中文）解讀：

- 這是一個很強的負面結果：某些「組合」的輔助目標 **不穩定**，可能造成大幅退步。
- 這支持我們的策略：「先快速 screening、一次只調一個旋鈕、不要過度宣稱」。

---

## 5) Negative compare result (teachers): Stage-A validation does not predict hard-gate robustness

（繁體中文）5）Teacher 的負面比較結果：Stage-A validation 無法預測 hard-gate 強健性

This is a key negative finding about evaluation design.

（繁體中文）這是關於評估設計的一個關鍵負面發現。

Summary table:

- `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`

（繁體中文）總表：

- `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`

Selected evidence (macro-F1):

（繁體中文）重點證據（macro-F1）：

- On `eval_only` (hard gate): macro-F1 is ~0.37–0.39
- On `expw_full` (cross-dataset): macro-F1 is ~0.37–0.41
- On `test_all_sources`: macro-F1 is ~0.62–0.65

Example rows (macro-F1):

（繁體中文）範例列（macro-F1）：

- `eval_only`:
  - RN18: 0.372670
  - B3:   0.392831
  - CNXT: 0.388980
- `expw_full`:
  - RN18: 0.374009
  - B3:   0.406649
  - CNXT: 0.382112

Interpretation:

- “Good in-distribution Stage-A validation” does **not** guarantee performance on mixed-domain / hard gates.
- This supports the final report’s evaluation design: separate training selection metrics from deployment-aligned gates.

（繁體中文）解讀：

- 「在 in-distribution 的 Stage-A validation 表現好」**不代表** 在 mixed-domain / hard gate 也會好。
- 這支持主報告的評估設計：訓練選擇指標與部署導向 gate 必須分開看。

---

## 6) “Result of the compare” — what I will show Prof. Lam (minimum set)

（繁體中文）6）「比較結果」— 我會在會議中給 Lam 教授看的最小集合

To keep the meeting concrete, I will focus on **three comparison blocks**:

（繁體中文）為了讓會議具體可討論，我會聚焦在 **三個比較區塊**：

1. **Deployment-facing webcam A/B (baseline vs adapted)**
   - the negative result is large and directly tied to real-world behavior
   - artifacts are clean and reproducible (same session replay)

   （繁體中文）**部署導向 webcam A/B（baseline vs adapted）**
      - 負面結果幅度大，且直接關聯實際使用行為
      - 產物乾淨可重現（同 session 重播）

2. **Offline student objective comparisons (KD/DKD vs NegL/NL variants)**
   - show that some auxiliary objectives regress macro-F1 / minority-F1
   - compare tables exist as stored markdown

   （繁體中文）**離線 student 目標函數比較（KD/DKD vs NegL/NL 變體）**
      - 顯示部分輔助目標會讓 macro-F1 / minority-F1 退步
      - compare table 已以 markdown 形式保存

3. **Teacher hard-gate robustness table**
   - shows why we must use hard gates (eval-only, ExpW) and cannot rely on Stage-A validation

   （繁體中文）**Teacher hard-gate 強健性表格**
      - 顯示為何必須使用 hard gate（eval-only、ExpW），不能只依賴 Stage-A validation

This gives a coherent “story”: domain shift is real, and naive or unstable objectives can harm both offline and deployment-facing results.

（繁體中文）這會形成一致的「故事線」：domain shift 是真實存在的，而天真的或不穩定的目標函數可能同時傷害離線與部署導向的結果。

---

## 7) Next-step plan (detailed, controlled, and evidence-driven)

（繁體中文）7）下一步計畫（詳細、可控、證據導向）

### 7.1 Priority A — Fix the webcam adaptation failure mode (one knob at a time)

（繁體中文）7.1 優先事項 A — 修正 webcam 適應的失敗模式（一次只調一個旋鈕）

Goal: identify whether the regression is caused by NegL policy, buffer quality, or conservative-tuning settings.

（繁體中文）目標：找出退步是由 NegL policy、buffer 品質，或保守式 tuning 設定造成。

Planned ablations (in order):

（繁體中文）預計的消融實驗（依序）：

1. **Self-learning positives only (NegL OFF)**
   - same buffer builder, but train only on high-confidence pseudo-label positives
   - expectation: if performance recovers, regression was likely due to NegL policy on medium-confidence frames

   （繁體中文）**只做 self-learning 正樣本（關掉 NegL）**
      - 使用同一個 buffer builder，但只用高信心 pseudo-label 正樣本訓練
      - 預期：若表現回升，退步可能主要來自中等信心幀的 NegL policy

2. **NegL-only vs mixed ratio sweep**
   - sweep the ratio of positives vs NegL-only frames
   - track acceptance rate and class distribution of buffer

   （繁體中文）**NegL-only 與混合比例掃描**
      - 掃描正樣本 vs NegL-only 幀的比例
      - 追蹤 buffer 的接受率與類別分佈

3. **Medium-confidence policy change**
   - compare current: `neg_label=<predicted_label>`
   - vs complementary-label selection (choose a plausible wrong class)
   - vs ignore medium confidence entirely

   （繁體中文）**中等信心 policy 調整**
      - 對比現行：`neg_label=<predicted_label>`
      - vs complementary-label（選一個合理但錯的類別）
      - vs 直接忽略所有中等信心樣本

4. **Multi-session evidence**
   - repeat the same protocol on multiple recorded sessions (different lighting/subject)
   - to avoid a conclusion based on a single session

   （繁體中文）**多 session 證據**
      - 在不同光照/不同人物的多段錄影 session 上重複相同協議
      - 避免結論只建立在單一 session

Acceptance criteria (must satisfy both):

（繁體中文）接受標準（必須同時滿足）：

- offline non-regression on eval-only and ExpW
- improved identical-session replay score (macro-F1 + stability/jitter) under fixed demo parameters

（繁體中文）

- eval-only 與 ExpW 離線不退步（non-regression）
- 在固定 demo 參數下，同一段 session 重播評分提升（macro-F1 + stability/jitter）

### 7.2 Priority B — Make paper-method comparison “supervisor-ready”

（繁體中文）7.2 優先事項 B — 讓論文方法比較達到「可給導師審閱」的狀態

We already have a protocol document defining fairness rules:

- `research/fair_compare_protocol.md`

（繁體中文）目前已經有定義公平規則的協議文件：

- `research/fair_compare_protocol.md`

Decisions needed from Prof. Lam:

（繁體中文）需要 Lam 教授協助決定的事項：

1. **Pick one “paper fairness anchor” dataset** (in addition to eval-only and ExpW)
   - suggested: RAF-DB basic (clean + common in FER papers)

   （繁體中文）**選一個「論文公平 anchor」資料集**（在 eval-only 與 ExpW 之外）
      - 建議：RAF-DB basic（相對乾淨且常見）

2. **Lock one full-budget schedule**
   - so that final tables are not mixed with screening runs

   （繁體中文）**鎖定一個 full-budget 訓練設定**
      - 讓最終表格不會混用 screening runs 與 full runs

3. **Decide minimal reproducibility requirement**
   - 1 seed vs 2 seeds for claims of “consistent effect”

   （繁體中文）**決定最小重現標準**
      - 需要 1 個 seed 或 2 個 seed 才能宣稱「效果一致」

### 7.3 Priority C — Convert negative results into thesis contribution

（繁體中文）7.3 優先事項 C — 把負面結果轉成論文/畢業論文的貢獻

Deliverables that explicitly present negative results as contribution:

（繁體中文）把負面結果明確當作貢獻的交付物：

- Ensure the main report and the negative result report are consistent and cite the same artifacts.
- Add a short “failure modes” subsection in the presentation:
  - preprocessing mismatch
  - BN running-stat drift
  - offline gate ≠ deployment improvement

（繁體中文）

- 確保主報告與負面結果報告一致，引用同一批 artifacts。
- 在簡報加入短的「失敗模式」小節：preprocessing mismatch；BN running-stat drift；offline gate ≠ deployment improvement（離線 gate 不等於部署改善）

## 8) Questions I want to ask Prof. Lam (so we align on scope)

（繁體中文）8）我想問 Lam 教授的問題（確認範圍與優先順序）

1. Should we prioritize **deployment success (webcam)** over **paper fairness comparison** for the next 2–3 weeks?
2. For the thesis, is it acceptable to frame the webcam adaptation track as:
   - “a controlled negative result + a rigorous evaluation protocol”,
   - even if we do not achieve a positive improvement by the deadline?
3. Which benchmark is preferred as the “paper anchor” dataset: RAF-DB basic vs FER2013 vs FERPlus vs AffectNet?
4. What is the expected standard of reproducibility for final claims: 1 seed vs 2 seeds?

（繁體中文）

1. 接下來 2–3 週，是否應該把 **部署成功（webcam）** 的優先度放在 **論文公平比較** 之前？
2. 對畢業論文而言，是否可以把 webcam adaptation 這條線定位為：
   - 「嚴格控制下的負面結果 + 嚴謹的評估協議」
   - 即使到期限前沒有達到正向提升？
3. 「paper anchor」資料集應選哪個：RAF-DB basic / FER2013 / FERPlus / AffectNet？
4. 最終結論的重現標準希望是 1 seed 還是 2 seeds？

---

## 9) Pointers (where to find the full details)

（繁體中文）9）索引（完整細節在哪裡）

- Main final report: `research/final report/final report version 2.md`
- Negative results report (detailed): `research/final report/negative result report/negative result report.md`
- Feb process log (contains exact Feb-21 A/B): `research/process_log/Feb process log/Feb_week3_process_log.md`
- Teacher hard-gate summary table: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
- Student compare tables (KD/DKD with NegL/NL variants): `outputs/students/_compare*.md`

（繁體中文）

- 主報告：`research/final report/final report version 2.md`
- 負面結果報告（更詳細）：`research/final report/negative result report/negative result report.md`
- 2 月流程紀錄（含 2/21 精準 A/B 數字）：`research/process_log/Feb process log/Feb_week3_process_log.md`
- Teacher hard-gate 總表：`outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
- Student compare tables（KD/DKD 與 NegL/NL 變體）：`outputs/students/_compare*.md`
