# 討論計劃（下一步）— 與導師會面

日期：2026-01-21

## 1) 會議目標（今日希望決定的事）

1) 確認未來 1–2 週的主要目標：
	- 部署準備（CPU 即時 demo 正確度 + FPS） vs
	- Domain shift robustness（ExpW / webcam-like） vs
	- 平衡（用最少工作同時 unblock 兩邊）

2) 同意一個**公平的 live 評估 protocol**，用嚟同 offline metrics 做對比。

3) 選擇下一個「最值得做」的 intervention track（只揀一條主線）：
	- Stabilization tuning（穩定化參數調整）
	- Pipeline parity fixes（live vs offline pipeline 對齊）
	- Domain-shift training（augmentation / long-tail）

## 2) 目前狀態（快速摘要）

- real-time demo 支援 device forcing：`--device {auto,cpu,cuda,dml}`。
- 已有 ExpW domain-shift baseline compare table，顯示有明顯 robustness gap。
- live scoring 已升級：同時計 `metrics.raw`、`metrics.smoothed` 同埋 `macro_f1_present`。

會議要展示的 artifacts：

- Next-step plan：`research/plan of next step/plan.md`
- ExpW domain-shift baseline table：`outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`
- Week log（決策+觀察）：`research/process_log/Jan process log/Jan_week4_process_log.md`

## 3) 核心問題（點解 live 會比 offline 差咁多）

觀察：

- 「Real-time macro-F1」經常睇落會比 offline evaluation 差好多。

可能成因（由最常見到較深層）：

1) **Metric artifact（live session 的評估假象）**
	- 好多 demo run 無手動 label → 根本計唔到 F1。
	- live session 經常只出現 1–2 種表情 → 用「7-class macro-F1」會被拉得好低。
	- 建議 live 用 `macro_f1_present`（只對 session 真正出現過的 class 做 macro）。

2) **Stabilization 改變咗輸出**
	- demo 的 `pred_label` 係穩定化後 label（EMA / vote / hysteresis）。
	- offline evaluation 用 raw logits → 兩者本身就唔係同一個輸出，唔可以直接比。

3) **Pipeline parity / domain shift**
	- face detector crop policy、resize、normalization、CLAHE、RGB/BGR 都可以造成 distribution shift。
	- webcam/ExpW 會有：光照、blur、compression、角度等差異。

4) **Regression issue（offline：KD/DKD 對比 CE）**
	- Interim report（2025-12-25）顯示：CE macro-F1 = 0.741952，而 KD/DKD 在 first run 未能 surpass CE。
	- KD/DKD 對 calibration 有提升（temperature-scaled ECE 約 0.027 vs CE 約 0.050），但 raw accuracy / macro-F1 反而下降。
	- 代表現時 KD/DKD 設定或訓練長度可能係「用 calibration 換咗 raw macro-F1」。

## 4) 決策規則（用一個 live 實驗決定下一步）

用 1 次有手動標註的 live run，並用 `--pred-source both` 評分：

- 如果 `metrics.raw` 明顯高過 `metrics.smoothed`：
	- 先調 stabilization（唔使 retrain，ROI 高）。

- 如果兩者都低，但 offline evaluation 其實唔差：
	- 先修 pipeline parity（crop/normalize/detector policy）再 retrain。

- 如果兩者都低，而且 ExpW 都低：
	- 進入 domain-shift training（augmentation → long-tail → target-aware fine-tune）。

## 5) 未來 3–7 日行動（具體 checklist）

### A) 收集公平的 live baseline（必做）

目標：一段 2–3 分鐘、CPU 強制、刻意做 4–5 種表情並手動標註的 session。

Commands：

```powershell
python demo/realtime_demo.py --model-kind student --device cpu
python scripts/score_live_results.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --out demo/outputs/<run_stamp>/score_results.json --pred-source both
```

Deliverable：

- 產生 `demo/outputs/<run_stamp>/score_results.json`（包含 `macro_f1_present` + raw/smoothed 比較）。

### B) Stabilization tuning（只在 raw > smoothed 時做）

目標：減少 over-smoothing，但保持 demo 穩定。

做法：

- sweep 2–3 組設定（小 grid）：EMA alpha / vote window / hysteresis threshold。
- 每組都重新 score，對比：
	- `metrics.smoothed.macro_f1_present` vs baseline
	- flip-rate / unstable segments（如果 scorer 有報）

Deliverable：

- 一個小表格：stability vs correctness trade-off。

### C) Pipeline parity checks（只在 live raw 異常低時做）

Checklist：

- 確認 face crop size 同 preprocessing 是否同 training/eval 一致。
- 檢查 color space + normalization。
- 檢查 CLAHE 使用一致性（on/off + 參數）。
- 如有需要：save 幾張 demo crops，再用 offline evaluator 跑，分離「pipeline 問題」定「model 問題」。

Deliverable：

- 一段短 note：pipeline mismatch found / not found + 改咗咩。

### D) Domain shift 改善（ExpW-first）

如果 A/B/C 做完仍然 ExpW 弱：

1) robustness augmentations（photometric + blur + compression）
2) long-tail 改善（只揀一個）：class-balanced loss 或 focal 或 logit adjustment
3) target-aware fine-tuning（ExpW-heavy mix，可選 KD/DKD）

Deliverable：

- 新的 compare table（`outputs/evals/`），顯示 minority-F1 有提升。

## 6) 想問導師的問題（用嚟 unblock 決策）

1) 下一個 milestone：更重視 CPU demo 正確度，定 ExpW robustness？
2) live KPI 建議報咩：accuracy / macro-F1-present / minority-F1-present？
3) demo 可接受的 stability vs responsiveness trade-off 係點？
4) training budget（GPU hours）同可接受複雜度（新 loss vs 只加 augmentation）有無限制？

5) 部署角度：偏向 **更高 raw macro-F1（CE-like）** 定 **更好 calibration（KD/DKD-like）**，定係兩樣都要？

6) 部署驗收門檻（建議例子）：
	- CPU FPS target（例如 >= 15/20/25 FPS）？
	- end-to-end latency（ms）同 flip-rate 上限？
	- 需唔需要同時報 stability metrics（flip-rate、dwell time）？

7) demo 最終輸出應該係：
	- raw logits argmax，定 stabilized `pred_label`，定係兩者都保留（UI toggle）？
	- 需唔需要加「不確定 / unknown」狀態，用 calibration 後 confidence threshold 去 gate？

8) domain shift：主力 target 應該用 ExpW-full 定 ExpW-HQ？可以接受幾多 in-domain accuracy 換 robustness？

9) reproducibility / artifact policy：要唔要固定一個「deployment checkpoint」同一個「research checkpoint」，並用 provenance（seed、manifests、preprocessing flags）避免 silent regression？

## 7) 風險 + 對策

- 風險：live labeling 唔一致 / 太短 → 指標唔穩。
	- 對策：統一 2–3 分鐘 protocol，cover 4–5 emotions。

- 風險：ExpW 改善咗，但真實 webcam 仍然唔得。
	- 對策：整一個小型「webcam-mini」標註 set，同 ExpW 一齊 track。

- 風險：未搞掂 pipeline parity 就開始 retrain，浪費時間。
	- 對策：嚴格跟 decision rule（raw vs smoothed vs offline）。

## 8) 需要的資源

- 30–60 分鐘：做一次有標註 live session + 2–3 次 stabilization sweep。
- 若批准訓練：1–2 次受控 run 的 GPU 時間（先 augmentation，再試 1 個 long-tail variant）。
