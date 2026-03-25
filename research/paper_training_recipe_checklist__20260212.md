# Paper training/protocol check (hyperparameter gap analysis)
Date: 2026-02-12

Goal: explain why our FER results can differ from paper-reported results **even after** matching the official split and reporting ten-crop, and convert each paper’s stated protocol into a concrete FC3 reproduction checklist.

Important scope note:
- “Paper gap” is often a combination of **(A) evaluation protocol** + **(B) preprocessing/alignment** + **(C) model capacity** + **(D) training hyperparameters**.
- This document focuses on (D) **hyperparameters/settings** and the closely related (A)/(B) settings that are frequently bundled together in papers.

Primary local evidence sources:
- Paper full text extracts: outputs/paper_extract/*.txt
- AffectNet Table-9 extraction (contains training recipe text): outputs/paper_extract/affectnet__table9__raw_pages.tsv
- Our protocol-aware FER2013 official summary (single-crop + ten-crop): outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md

---

## 1) FER2013 SOTA paper (VGGNet; ten-crop)
Paper: “Facial Emotion Recognition State of the Art Performance on FER2013”
- Local extract: outputs/paper_extract/Facial Emotion Recognition State of the Art Performance on FER2013.txt

What the paper states (protocol + training settings):
- Split: **official** train/val/test (ICML).
- Training augmentation (explicit):
  - random rescaling ±20%, shift (H/V) ±20%, rotation ±10°, each applied with probability 50%
  - then “ten-cropped” to **40×40**
  - random erase on each crop with p=50%
  - normalization: divide pixels by 255
- Training length: **300 epochs** optimizing cross-entropy.
- Fixed momentum **0.9**, weight decay **1e-4**.
- Evaluated using validation accuracy; tested using **standard ten-crop averaging**.
- They also mention additional tuning: cosine annealing; combine training + validation to reach 73.28%.

Why this can explain a big gap vs our system:
- Input/crop scale is materially different (paper uses 40×40 crop pipeline; our student/teacher pipelines are generally img224 + ImageNet normalization).
- They “tune hyperparameters thoroughly” (optimizer + LR scheduler) and report 300-epoch training; if our student was not trained on FER2013-only with a similar schedule, the comparison is not controlled.

FC3 checklist (what to match if we try to reproduce in this repo):
- Use the official FER2013 split (already present as manifests under Training_data/FER2013_official_from_csv/).
- Add a FER2013-only training config:
  - grayscale vs RGB decision (paper extract doesn’t clearly force grayscale; confirm by opening the PDF)
  - input size/crop policy: reproduce 40×40 ten-crop augmentation during training
  - normalization: /255 instead of ImageNet mean/std
  - training length ~300 epochs, SGD (and/or the optimizer variants they list), momentum=0.9, wd=1e-4
  - LR scheduler: include cosine annealing variant
- Keep *selection* fair: choose best by **validation**, not by PublicTest/PrivateTest.

---

## 2) “Benchmarking Deep FER” protocol paper (Keras; uniform protocol across models)
Paper: “Benchmarking Deep Facial Expression Recognition: An Extensive Protocol with Balanced Dataset in the Wild”
- Local extract: outputs/paper_extract/Benchmarking Deep Facial Expression Recognition An Extensive Protocol with Balanced Dataset in the Wild.txt

What the paper states (hyperparameters/settings):
- Data augmentation mentions include: width shift (0.2), height shift (0.2), zoom range (0.2), horizontal flip True.
- Uniform hyperparameters applied to all models:
  - learning rate 1e-4
  - LR reduction on validation loss: patience 10 epochs, factor 0.5, min LR 1e-10
  - early stopping: patience 20 epochs
  - training strategy: pre-trained weights; freeze layers; add new layers; train “for 30 epochs or until early stopping”; then unfreeze and continue with early stopping
- Their “Hyperparameters” list includes Beta_1=0.9 and Beta_2=0.999 (typical Adam-family settings), implying Adam/AdamW-style optimizer.

Why this can explain gaps across papers (author-to-author differences):
- This paper’s recipe is a “Keras fine-tune protocol” (freeze/unfreeze + early stopping + reduce-on-plateau), which is a very different regime than (e.g.) 300-epoch SGD on FER2013.

FC3 checklist:
- If we compare to numbers from this paper, we should mirror:
  - augmentation strength (shift/zoom/flip)
  - early-stopping + reduce-on-plateau schedule
  - pretrained backbone and freeze/unfreeze fine-tuning phases

---

## 3) Face-regions paper (padding; Adam; high LR)
Paper: “Expression Analysis Based on Face Regions in Read-world Conditions”
- Local extract: outputs/paper_extract/Expression Analysis Based on Face Regions in Read-world Conditions.txt

What the paper states (hyperparameters/settings + preprocessing):
- Optimizer: Adam.
- Learning rate: initial 0.05, and “smaller learning rate will be utilized” if testing accuracy decreases.
- Augmentation: random crop to square + random horizontal flip.
- Normalization: same mean/variance per channel.
- Max epoch: 100, early stopping.
- They train the system five times and “choose the best model according to the performance on the testing set”.
- Testing: replace training augmentation with center-crop around the center.
- Padding: they convert non-square crops to square by padding; they compare “padding vs non-padding”.

Important comparability warning:
- Selecting the best model **based on test-set performance** is not a strict evaluation protocol; it can inflate reported test accuracy.

FC3 checklist:
- If we want to be fair and still learn from this paper:
  - replicate padding/crop policy
  - replicate augmentation/optimizer schedule
  - but choose best checkpoint by validation, not by test

---

## 4) AffectNet paper (training recipe + crop details)
Paper: “AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild”
- Local extract: outputs/paper_extract/AffectNet A Database for Facial Expression, Valence, and Arousal Computing in the Wild.txt
- Training recipe text appears in: outputs/paper_extract/affectnet__table9__raw_pages.tsv

What we can extract from the local Table-9 pages (training settings + preprocessing):
- Training length: 20 epochs.
- Batch size: 256.
- Base learning rate: 0.01, decreased step-wise by factor 0.1 every 10,000 iterations.
- Momentum: 0.9.
- Augmentation mentions: five crops of 224×224 and their horizontal flips.
- Face crop/resize mentions: faces cropped and resized to 256×256 pixels; “No facial registration” is mentioned in the extracted pages.

FC3 checklist:
- For comparisons involving AffectNet-like settings, pay attention to:
  - whether the evaluation is original skewed test vs balanced subsets
  - crop/resize policy (256 base, 224 crop) and “no registration”
  - SGD+momentum schedule with step-wise LR decay

---

## 5) Masked-Augmented FERPlus paper (oversampling + ImageDataGenerator config)
Paper: “Facial_Emotion_Recognition_Using_Masked-Augmented_FERPlus”
- Local extract: outputs/paper_extract/Facial_Emotion_Recognition_Using_Masked-Augmented_FERPlus.txt

What the paper states (hyperparameters/settings):
- Augmentation library: Keras ImageDataGenerator.
- Augmentations: rotation, stretching, shifting.
- Explicit ImageDataGenerator config (Table III):
  - rotation_range=30
  - width_shift_range=0.1
  - height_shift_range=0.1
  - shear_range=0.1
  - zoom_range=0.1
  - horizontal_flip=True
  - fill_mode='nearest'
- Class imbalance handling via oversampling; then mask injection:
  - add masked faces equal to 25% of each class size to keep balance
  - mentions MaskTheFace (github.com/aqeelanwar/MaskTheFace)
- Models tried include CNN, VGG16, ResNet, InceptionV3 (exact training hyperparams for each model may require PDF verification).

FC3 checklist:
- If we want to reproduce the “masked augmentation” idea:
  - implement mask overlay augmentation (probability + per-class balancing)
  - replicate ImageDataGenerator-like transforms in PyTorch
  - evaluate on FERPlus with an explicitly stated label-space mapping

---

## 6) RAF-AU paper (alignment + grayscale; mostly AU)
Paper: “RAF-AU Database In-the-Wild Facial Expressions with Subjective Emotion Judgement and Objective AU Annotations”
- Local extract: outputs/paper_extract/RAF-AU Database In-the-Wild Facial Expressions with Subjective Emotion Judgement and Objective AU Annotations.txt

What the paper states (preprocessing relevant to why papers differ):
- Alignment: register images to a reference face using affine transformation based on 5 landmarks.
- Crop: 100×100.
- Convert to grayscale.

Relevance to our FER work:
- This is not a 7-class FER accuracy paper, but it highlights a common “hidden variable”: landmark-based alignment + grayscale preprocessing.

---

## Summary: is the paper gap caused by hyperparameters?
Yes, it can be a major contributor.

From the extracted text, the papers differ drastically in:
- input resolution and crop policy (40×40 vs 224/256)
- optimizer family and schedule (SGD+momentum vs Adam; cosine annealing vs reduce-on-plateau)
- augmentation strength and style
- early stopping vs fixed long schedules
- selection protocol (some select best by test)

Because these are high-impact settings, two systems can be “on the same dataset” and still differ by a large margin.

---

## Actionable next step for this repo (lowest-risk improvement path)
1) Choose **one anchor target** (FER2013 73.28% paper), and reproduce its settings as a *controlled* FC3 experiment:
   - match crop policy and normalization
   - match optimizer/schedule + epoch budget
   - keep official split + ten-crop reporting
2) Add a small hyperparameter sweep (LR × weight decay × augmentation strength) on FER2013-only training.
3) Only after closing the most obvious setting gaps, revisit “method” improvements (NegL/self-learning) so we don’t mistake protocol drift for method effect.
