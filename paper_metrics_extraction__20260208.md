# Paper Metrics / Protocol Extraction Notes
Date: 2026-02-08

This file records **quotable** protocol/metric statements pulled from local extracted text under `outputs/paper_extract/`.

## FER2013 — “Facial Emotion Recognition State of the Art Performance on FER2013”
Source extract: `outputs/paper_extract/Facial Emotion Recognition State of the Art Performance on FER2013.txt`

Key metric claim:
- “state-of-the-art single-network accuracy of **73.28 %** on FER2013 without using extra training data.”

Protocol constraints (for fair comparison):
- “adhere to the **official training, validation, and test sets** as introduced by the ICML.”
- “tested using **standard ten-crop averaging**.”

Implication:
- We should not compare our `test_fer2013_uniform_7` numbers to this paper unless we evaluate on the ICML/Kaggle official split.

## RAF-DB / FER+ / ExpW — “Expression Analysis Based on Face Regions in Read-world Conditions”
Source extract: `outputs/paper_extract/Expression Analysis Based on Face Regions in Read-world Conditions.txt`

Table 5 (RAF-DB padding vs non-padding):
- Whole face accuracy: non-padding **77.31**, padding **82.69**.

Table 6 (whole face accuracies on testing set):
- FER+ **81.93**
- RAF-DB **82.69**
- ExpW **71.90**

Notes:
- The extracted text around Table 6 is slightly line-wrapped; verify in PDF if you need perfect table alignment.

## AffectNet — “AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild”
Source extract: `outputs/paper_extract/AffectNet A Database for Facial Expression, Valence, and Arousal Computing in the Wild.txt`

From Table 7 (per-class F1, Weighted-Loss approach), a derived macro-F1 note exists:
- `outputs/paper_extract/affectnet__table7_weightedloss_macro_f1.md`
- Derived macro-F1 (Top-1, skew-normalized): **0.625**

## FERPlus masked augmentation — “Facial_Emotion_Recognition_Using_Masked-Augmented_FERPlus”
Source extract: `outputs/paper_extract/Facial_Emotion_Recognition_Using_Masked-Augmented_FERPlus.txt`

From abstract (quoted numbers):
- “accuracy on the original dataset, which was **0.752**, slightly surpassed its performance on the masked augmented dataset, which achieved an accuracy of **0.747**.”

## Benchmarking protocol paper — “Benchmarking Deep Facial Expression Recognition: An Extensive Protocol with Balanced Dataset in the Wild”
Source extract: `outputs/paper_extract/Benchmarking Deep Facial Expression Recognition An Extensive Protocol with Balanced Dataset in the Wild.txt`

What we can safely extract from the text dump:
- The paper reports performance tables (Table 2/3) with validation accuracy and cross-dataset test accuracy on their BTFER dataset.
- Table dumps in the extracted text are flattened/concatenated; use PDF viewing if you need a specific model’s exact number.

## RAF-AU paper
Source extract: `outputs/paper_extract/RAF-AU Database In-the-Wild Facial Expressions with Subjective Emotion Judgement and Objective AU Annotations.txt`

Notes:
- This paper primarily reports AU detection metrics (AUC-ROC, F1) rather than 7-class FER accuracy.
- It is still relevant background for why RAF-style datasets can be used for expression recognition in the wild.
