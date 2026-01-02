# Real-time-Facial-Expression-Recognition-System
Full file download:
https://drive.google.com/file/d/1Uta_bAuvfSNi71V2hDtq18YQE7QwgaZT/view?usp=sharing

Real-time Facial Expression Recognition System — Project Description
The Real-time Facial Expression Recognition System is a deployment‑oriented FER project designed to classify seven basic emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from live video streams with low latency, stable predictions, and reliable confidence scores on standard consumer hardware.
Rather than focusing solely on offline accuracy, this project addresses the entire practical pipeline—from data quality and model training to knowledge distillation, calibration, and real-time deployment. It specifically tackles real‑world challenges such as class imbalance, cross‑dataset domain shift, confidence miscalibration, and the gap between offline and real-time performance.
Data Pipeline
The system integrates multiple in‑the‑wild FER datasets (FER2013, FERPlus, RAF‑DB, AffectNet, ExpW, etc.) under a unified 7‑class label space. A full cleaning and manifest‑validation workflow ensures consistent paths, labels, and dataset integrity. A mixed‑source test set (~49k samples) is used to evaluate robustness beyond a single curated dataset.
Modeling Approach
A teacher–student framework is adopted:
- Teachers: ResNet18, EfficientNet‑B3, ConvNeXt‑Tiny
- Ensemble: Weighted logit fusion to generate high‑quality soft labels
- Student: MobileNetV3‑Large for real-time efficiency
Training follows a CE → KD → DKD pipeline, with temperature scaling applied to improve confidence calibration for downstream thresholding and selective prediction.
Evaluation
Beyond accuracy, the project emphasizes Macro‑F1, per‑class F1, NLL, and ECE to reveal long‑tail weaknesses in minority classes such as Fear and Disgust.
Findings show:
- CE student achieves the highest accuracy
- KD/DKD students achieve significantly better calibration
This makes the distilled models more suitable for real-time applications requiring trustworthy confidence scores.
Deployment
The real-time system integrates YuNet for fast face detection, CLAHE for preprocessing consistency, and temporal smoothing to reduce prediction flicker. The final demo runs smoothly on a standard laptop and provides stable, interpretable emotion outputs.
Outcome
The project delivers:
- A validated multi‑source FER dataset pipeline
- Three high‑performance teacher models
- A distilled, real-time‑ready student model
- Calibrated confidence outputs
- A fully functional real-time FER demo
This work demonstrates a complete research‑to‑deployment workflow and lays the foundation for future extensions such as Negative Learning, meta‑learning, and more advanced distillation strategies.


Real-time Facial Expression Recognition System 是一個以實際部署為導向的即時表情識別系統，目標是在一般電腦硬件上，以低延遲、穩定且可靠的方式從即時影像中辨識七種基本情緒（Angry、Disgust、Fear、Happy、Sad、Surprise、Neutral）。本專案並非單純追求離線準確率，而是完整處理從資料品質、模型訓練、知識蒸餾、校準到即時部署的整條技術鏈，並特別針對「資料不平衡」、「跨資料來源的 domain shift」、「模型信心校準」與「離線與即時表現落差」等實務問題提出解決方案。
在資料層面，本專案整合多個 in‑the‑wild FER 資料集（FER2013、FERPlus、RAF‑DB、AffectNet、ExpW 等），並建立統一的 7 類別標準、清洗流程與完整 manifest 驗證機制，確保所有影像路徑、標籤與來源一致無誤。透過混合來源的測試集（n≈49k），本專案能更準確評估模型在真實場景下的穩健性，而非依賴單一資料集的理想化結果。
在模型層面，本專案採用 teacher–student 架構：先以 ResNet18、EfficientNet‑B3、ConvNeXt‑Tiny 訓練高性能教師模型，再以加權 ensemble 融合其 logits，並輸出 soft labels 供學生模型學習。學生模型選用 MobileNetV3‑Large，以兼顧效能與即時性。訓練流程包含 CE → KD → DKD 三階段，並以溫度縮放（Temperature Scaling）改善模型信心校準，使其在即時應用中更可靠。
在評估層面，本專案不僅報告 Accuracy，也重視 Macro‑F1、Per‑class F1、NLL、ECE 等指標，以揭示長尾類別（如 Fear、Disgust）在不平衡資料下的真實表現。結果顯示：CE 學生模型在準確率上最佳，而 KD/DKD 在校準指標上更優，適合需要可靠信心分數的應用。
在部署層面，系統整合 YuNet 進行即時人臉偵測，並加入 CLAHE、影格平滑（temporal smoothing）等技術，以減少預測跳動並提升使用者體驗。整體系統可在一般筆電上以即時速度運行，並提供穩定的表情輸出。
本專案最終成果包括：完整的資料清洗與驗證流程、三個教師模型、經 KD/DKD 訓練的學生模型、校準後的可靠信心輸出、以及可即時運行的表情識別 Demo。此系統展示了從研究到實作部署的完整流程，並為未來加入負學習（Negative Learning）、meta‑learning 或更高效的蒸餾策略奠定基礎。

