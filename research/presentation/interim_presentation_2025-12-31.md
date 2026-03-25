# Real-time Facial Expression Recognition (FER) — Interim Presentation

> **Replace placeholders**: `YOUR_NAME`, `SID`, `SUPERVISOR`, `SEMESTER`.
> 
> Data in this deck is pulled from current repo artifacts (teachers / ensemble / students) and the Week 5 process log.

---

## 🎤 Slide 1 — Title

**內容：**
- **Project Title**: Real-time Facial Expression Recognition (FER) System
- **Your Name + SID**: Ma Kai Lun Donovan 24024192D
- **Supervisor**:  Prof LAM Kin Man
- **Semester**: SEMESTER

**你講：**
> 「我嘅 project 係 real-time facial expression recognition，
> 今日會講下背景、方法、實驗結果同埋 demo。」

---

## 🎤 Slide 2 — Motivation & Problem

**內容：**
- Real-time FER 嘅重點唔止係 offline accuracy
- 主要挑戰
  - **Domain shift**：dataset vs webcam 光線/角度/畫質
  - **Class imbalance**：少數類別（例如 Fear/Disgust）較難
  - **Calibration**：confidence 要可信（唔好「自信爆棚」但錯）
  - **Latency**：要即時，唔可以太慢

**你講：**
> 「Real-time FER 最大嘅難度唔係 offline accuracy，而係 domain shift、class imbalance 同 calibration。
> Offline 模型好準，但 webcam 一開就跌分，所以我嘅目標係做一個 準、快、穩定 嘅 real-time system。」

---

## 🎤 Slide 3 — Project Goal

**內容：**
- **Input** → webcam frame
- **Output** → 7 expressions
  - Angry / Disgust / Fear / Happy / Sad / Surprise / Neutral
- **Constraints**
  - low latency
  - stable predictions (less flicker)
  - calibrated confidence

**你講：**
> 「系統要做到：
> 1. 7-class expression
> 2. real-time latency
> 3. prediction 唔好 flicker
> 4. confidence 要可信（calibration）」

---

## 🎤 Slide 4 — Dataset & Challenges (New Data)

**內容：**
- 多 dataset source（project used / cleaned）
  - FERPlus, RAF-DB, AffectNet, ExpW (and others)
- **主要 training manifest**：`Training_data_cleaned/classification_manifest.csv`
- **Class imbalance（counts, cleaned manifest, N=466,284）**

| Class | Count |
|---|---:|
| Neutral | 94,327 |
| Happy | 88,168 |
| Sad | 64,066 |
| Surprise | 58,913 |
| Angry | 55,392 |
| Disgust | 54,439 |
| Fear | 50,979 |

- **Mixed-source benchmark**：`Training_data_cleaned/test_all_sources.csv` (N=48,928)

**你講：**
> 「Dataset 有 imbalance，尤其係 Fear/Disgust/Angry 呢啲會難啲。
> 如果唔處理，模型會偏向 Happy/Neutral。
> 所以我用 **macro-F1**（每一類平均）做主要 metric，唔畀大類別『食晒』個分數。」

---

## 🎤 Slide 5 — Pipeline Overview

**內容：**
- Teacher → Ensemble → Softlabels → Student → Real-time demo

```mermaid
flowchart LR
  A[Clean & unify datasets\n(manifests)] --> B[Train teacher models\n(RN18 / B3 / CNXT)]
  B --> C[Evaluate on mixed-source benchmark\n(test_all_sources)]
  C --> D[Ensemble selection\n(weighted logits)]
  D --> E[Export softlabels\n(for KD/DKD)]
  E --> F[Train student\nCE → KD → DKD]
  F --> G[Real-time demo\nYuNet + CLAHE + Student]
```

**你講：**
> 「整個 pipeline 係：
> 1. Clean dataset
> 2. Train 3 teachers
> 3. 用 mixed-source benchmark 揀 ensemble
> 4. Export softlabels
> 5. Student 做 CE → KD → DKD
> 6. 最後落到 real-time demo」

---

## 🎤 Slide 6 — Teacher Models (New Data)

**內容：**
- Teachers (Stage A @224)
  - RN18 (ResNet-18)
  - B3 (EfficientNet-B3)
  - CNXT (ConvNeXt-Tiny)
- **Teacher validation performance (macro-F1)**

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| RN18 | 0.7862 | 0.7808 |
| B3 | 0.7961 | 0.7910 |
| CNXT | 0.7941 | 0.7890 |

- Observation: Happy/Neutral generally strongest; minority classes (Disgust/Fear/Angry) lower

**你講：**
> 「三個 teacher 都係 strong baseline。
> B3 同 CNXT 整體最好，RN18 對某啲 class（例如 Fear）表現都唔差。
> 因為佢哋互補，所以我用 ensemble 去提升穩定性同泛化。」

---

## 🎤 Slide 7 — Ensemble Performance (New Data)

**內容：**
- **Weighted ensemble (logit-level)**
  - Weights: RN18 / B3 / CNXT = **0.4 / 0.4 / 0.2**
- **Mixed-source benchmark**: `test_all_sources` (N=48,928)
- Ensemble metrics (domain shift observed)

| Metric | Value |
|---|---:|
| Accuracy | 0.6873 |
| Macro-F1 | 0.6596 |
| ECE | 0.2877 |

**你講：**
> 「Ensemble 喺 clean validation 係好準，但去到 mixed-source benchmark 就跌分。
> 呢個係典型 domain shift：dataset 同 webcam 真實環境唔同。
> 所以我下一步用 teacher-student（softlabels）訓練 student，目標係更 robust。」

---

## 🎤 Slide 8 — Student Results (CE / KD / DKD) (New Data)

**內容：**
- Student model: MobileNetV3-Large (fast for real-time)
- Key idea:
  - **CE**: best raw accuracy baseline
  - **KD/DKD**: improves robustness & calibration
  - **Temperature scaling** for calibrated confidence

**Student performance (validation)**

| Training | Acc | Macro-F1 | TS ECE | TS NLL |
|---|---:|---:|---:|---:|
| CE | 0.7502 | 0.7420 | 0.0499 | 0.7778 |
| KD (baseline, 5ep) | 0.7284 | 0.7266 | **0.0271** | 0.7839 |
| DKD (+5ep from KD) | 0.7357 | 0.7368 | 0.0348 | 0.7835 |

**Extra ablation (NegL / NL)**
- KD + NegL (entropy gate): Macro-F1 0.7198, TS ECE 0.0398 (no gain vs KD baseline)
- DKD + NegL: Macro-F1 0.7348, TS NLL 0.7926 (slightly worse vs DKD baseline)
- NL(proto) sweep is stable but nearly inactive (applied_frac < 0.03%) → needs redesign (use richer features)

**你講：**
> 「Student CE accuracy 最高，但 KD/DKD 嘅 calibration 會更好。
> Real-time 系統最驚就係『好有信心但錯』，所以我會用 temperature scaling。
> 另外，NegL 喺目前設定未見到明顯提升；NL(proto) 係穩定，但太少觸發，所以要改 feature source。」

---

## 🎤 Slide 9 — Demo System Architecture

**內容：**
- Face detection: **YuNet** (ONNX)
- Preprocessing: **CLAHE** (contrast normalization)
- Inference: **Student model** (MobileNetV3)
- Output: expression label + confidence
- Next add-on: temporal smoothing / thresholding (reduce flicker)

**你講：**
> 「Real-time pipeline 已經跑得郁：detect + classify 都做到。
> 下一步係 tune smoothing 同 threshold，令 label 唔好跳來跳去。」

---

## 🎤 Slide 10 — Demo (Interim Version)

**內容：**
- Put screenshot / short GIF here
- Current status:
  - detection OK
  - classification OK
  - tuning in progress (smoothing + calibration + thresholds)

**Screenshot placeholder**
- (Insert from your demo run output / screen recording)

**你講：**
> 「而家 demo 可以 detect 到 face 同埋出 expression label。
> 準確度仲可以再 tune，但 interim version 係可用同可展示。」

---

## 🎤 Slide 11 — Limitations

**內容：**
- Domain shift (webcam lighting / pose / blur)
- Minority class performance (Fear/Disgust still harder)
- Real-time jitter / flicker (frame-to-frame instability)
- Calibration under noise (confidence drift)

**你講：**
> 「現階段最大問題係 minority class、webcam noise 同 real-time jitter。
> 依家可以跑，但要再 refine 先會更穩定。」

---

## 🎤 Slide 12 — Next Steps (Based on New Data)

**內容：**
- Real-time smoothing (EMA / hysteresis / majority vote window)
- Calibration improvements
  - per-class temperature / class-wise thresholds
- NL + NegL research direction
  - NegL: do a parameter sweep (lower weight/ratio, adjust gate)
  - NL(proto): change prototype source to **penultimate features** (current logits-proto too “easy”)
- Hard-sample mining / error analysis (webcam-like cases)
- Deployment optimization
  - ONNX runtime / INT8 quantization (latency)

**你講：**
> 「之後會做 NL + NegL 去改善 minority class 同 calibration，
> 再做 real-time smoothing 同 quantization，令佢更快更穩。」

---

## 🎤 Slide 13 — Conclusion

**內容：**
- Teacher–student pipeline 已經完成
- Ensemble + student 已經 workable
- Demo pipeline 已經跑得郁
- 下一步：提升 real-time 穩定性 + calibration + minority class

**你講：**
> 「整體 pipeline 已經完成，demo 已經跑得郁。
> 下一步係提升 real-time 穩定性同 calibration，令佢更貼近真實 webcam 場景。」
