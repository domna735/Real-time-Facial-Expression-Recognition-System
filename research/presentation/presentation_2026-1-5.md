
# Real-time Facial Expression Recognition (FER) — Presentation (2026-01-05)

Author: Ma Kai Lun Donovan (24024192D)  
Supervisor: Prof. LAM Kin Man  
Institution: The Hong Kong Polytechnic University (PolyU)  

Notes:
- This file is a PowerPoint-ready script in Markdown.
- Add **page number** on every slide in PPT.
- Add **PolyU logo** (top-right recommended).
- Suggested font sizes in PPT: Title 36–40, Body 20–24, Caption 16–18.

---

## Slide 1 / 19 — Title

**Real-time Facial Expression Recognition (FER): Teacher–Student Distillation Pipeline and Deployment**

**Deliverables**
- A reproducible training pipeline (data → teachers → soft labels → student)
- A real-time inference demo (face detection + preprocessing + student classifier)

**Speaker notes (English)**
- Hi everyone. Today I’ll present my real-time facial expression recognition system.
- The core idea is a teacher–student pipeline: we train strong teacher models offline, export soft labels, and distill that knowledge into a small student model that can run in real time.
- I’ll first explain the “why” and the theory—what baseline and pipeline mean, why distillation works, and what CE/KD/DKD are optimizing.
- Then I’ll show the real results already saved in this repo, and finish with the next evaluation step: domain shift, with minority-F1 as the primary metric.

---

## Slide 2 / 19 — Outline

1) Problem and constraints (real-time + robustness)  
2) Definitions: baseline and pipeline  
3) Dataset foundation (cleaned unified manifests)  
4) Distillation theory: what knowledge is transferred  
5) Distillation mechanics: temperature, ensembles, failure modes  
6) Teacher models (RN18 / B3 / ConvNeXt) and why  
7) Losses part 1: CE and KD  
8) Losses part 2: DKD (target vs non-target)  
9) Calibration: NLL, ECE, temperature scaling (TS)  
10) Student choice + CE/KD/DKD baseline results  
11) NegL and NL mechanisms (theory)  
12) Ablation summary + evidence interpretation  
13) Domain shift evaluation plan (minority-F1 focus)  
14) Real-time deployment architecture and next steps

**Speaker notes (English)**
- This is a 15-minute talk, so I’m keeping each concept tight and separated.
- I’ll start with the problem and the constraints, then define baseline and pipeline so the experiments make sense.
- Next I’ll explain distillation and the loss functions—CE, KD, and DKD—and also calibration, because confidence matters in a real-time demo.
- After that, I’ll show the actual teacher and student results from this repository, and we’ll interpret what the NL/NegL experiments really say.
- Finally, I’ll propose the domain-shift evaluation that can give stronger evidence.

---

## Slide 3 / 19 — Problem Definition (What we want to solve)

**Task**
- 7-class facial expression classification: Angry / Disgust / Fear / Happy / Sad / Surprise / Neutral

**Why this is hard in real-time**
- **Domain shift**: training datasets vs webcam (lighting, pose, blur, background)
- **Class imbalance**: minority expressions (e.g., Fear / Disgust) are harder and less frequent
- **Calibration**: confidence must be meaningful (avoid “high-confidence wrong”)
- **Latency constraints**: the final model must be fast enough for real-time feedback

**Speaker notes (English)**
- The task is simple to state: classify 7 expressions from a webcam stream.
- The hard part is that the webcam domain is messy—lighting changes, pose changes, blur, and background—so the test distribution is different from curated datasets.
- Also, the class distribution is not balanced. Some expressions are naturally rare and harder, like fear and disgust.
- So for deployment, I care about two things: does it fail badly on minority classes, and is it overconfident when it’s wrong?
- That’s why the whole design emphasizes robustness and calibration, not just headline accuracy.

---

## Slide 4 / 19 — Key Definitions (Baseline vs Pipeline)

**Baseline (definition)**
- A baseline is the simplest credible reference system.
- Purpose: without a baseline, “improvements” cannot be interpreted.

**In this project, baselines mean**
- Same dataset and evaluation, but simplest training recipe (e.g., CE-only student, then KD baseline).

**Pipeline (definition)**
- A machine learning pipeline is the end-to-end workflow: data → training → evaluation → deployment.
- A clear pipeline improves **reproducibility** and enables **controlled comparisons**.

**Speaker notes (English)**
- Before the methods, I want to define two words I’ll use a lot.
- A baseline is basically the control group: the simplest credible system we compare against.
- If we don’t have that, we can’t honestly say a new method helped.
- A pipeline is the end-to-end workflow: data manifests, training, evaluation, and then deployment.
- The point is reproducibility: same dataset definition and same evaluation code, so small differences are meaningful.

---

## Slide 5 / 19 — Dataset Foundation (Real numbers from repo)

**Unified cleaned training manifest**
- File: `Training_data_cleaned/classification_manifest.csv`
- Total rows: **466,284**

**Source composition (rows by dataset source)**
- FER2013 (uniform 7): 140,000
- FERPlus: 138,526
- ExpW (full): 91,793
- AffectNet (full balanced): 71,764
- RAFDB-basic: 15,339
- RAFML (argmax): 4,908
- RAFDB-compound (mapped): 3,954

**Mixed-source benchmark**
- File: `Training_data_cleaned/test_all_sources.csv`
- Total rows: **48,928**

**Why this matters (theory)**
- Combining multiple sources increases diversity but also increases domain mismatch.
- Robustness and minority-F1 become more important than optimizing only accuracy.

**Speaker notes (English)**
- This slide is just to anchor everything in the actual data used in this repo.
- We build a unified cleaned manifest by merging multiple datasets into the same 7-class label space.
- The benefit is diversity: more faces, more conditions.
- The trade-off is that domain mismatch becomes larger—different datasets have different styles and biases.
- So later, it’s natural to test domain shift explicitly, for example evaluating on ExpW test.

---

## Slide 6 / 19 — Distillation Theory (What knowledge is transferred)

**Motivation**
- A real-time system needs a fast model (student), but high accuracy often requires larger models (teachers).

**Learning signal: one-hot vs soft targets**
- In standard supervised learning, the target is a one-hot label distribution $y$.
- In distillation, the target is a *teacher distribution* $p_t(\cdot\mid x)$ that contains extra information beyond the correct class.
- This extra information is often called **dark knowledge**: which wrong classes look “similar” for a given input.

**Why soft targets can help generalization (deeper intuition)**
- One-hot supervision only says “this class is correct”; it does not tell the model which mistakes are *less wrong*.
- Soft targets shape the decision boundary by providing *relative similarity structure* between classes.
- This can act like a **regularizer**: it discourages overly sharp/overconfident solutions that fit noise.

**Teacher vs Student**
- Teacher: higher capacity, slower, learns richer representations.
- Student: smaller and faster, suitable for real-time deployment.

**Distillation idea**
- Instead of learning only from one-hot labels, the student learns from **soft targets** produced by teachers.
- Soft targets encode inter-class relationships (e.g., Fear vs Surprise confusion), which can improve generalization.

**Speaker notes (English)**
- The motivation for distillation is practical: I want a small, fast model in the demo, but I can train bigger models offline.
- The theory point is: teachers don’t just output the correct label—they output a distribution.
- That distribution contains “dark knowledge”: which wrong classes look similar for this input.
- One-hot labels can’t express that. They only say “this is class y”.
- So distillation can shape a smoother decision boundary, and that can help generalization, especially when the domain shifts.

---

## Slide 7 / 19 — Distillation Mechanics (Temperature, ensembles, failure modes)

**Temperature $T$ (mechanics)**
- Distillation uses softened distributions:
$$p^T(k\mid x)=\mathrm{softmax}(z_k/T).$$
- Larger $T$ exposes non-top classes, increasing the “dark knowledge” signal.
- Practical note: gradients scale with $T$, so the KD term is often multiplied by $T^2$ to keep gradient magnitudes comparable.

**Why ensembles help (mechanics)**
- Ensemble soft labels average multiple teachers:
$$p_{ens}(\cdot\mid x)=\frac{1}{M}\sum_{m=1}^M p_{t_m}(\cdot\mid x).$$
- This reduces teacher variance and can smooth out model-specific errors.

**Failure modes (important for interpretation)**
- If teachers are biased under domain shift, the student can inherit that bias.
- If the student has insufficient capacity, it may underfit the teacher signal.

**Speaker notes (English)**
- Here’s the practical mechanics.
- Temperature $T$ controls how “soft” the teacher distribution is. Higher temperature reveals more information in the non-top classes.
- Next, we often use an ensemble of teachers, because averaging reduces variance and removes some model-specific mistakes.
- But distillation is not a guarantee. If teachers have a systematic bias under domain shift, the student can inherit it.
- That’s why later we don’t stop at in-domain validation—we plan domain-shift evaluation.

---

## Slide 8 / 19 — Teacher Models (What we used, and why)

**Teachers (Stage A @ 224)**
- ResNet-18 (RN18): efficient CNN baseline, strong for speed/robust features.
- EfficientNet-B3 (B3): good accuracy–efficiency trade-off, strong teacher candidate.
- ConvNeXt-Tiny (CNXT): modern convolutional features, complementary inductive bias.

**Why multiple teachers (theory)**
- Different architectures encode different inductive biases (e.g., feature hierarchies, receptive fields, normalization behavior).
- Using multiple teachers helps because it can reduce *model-specific errors*.
- Conceptually, teacher ensembling approximates **variance reduction**: averaging multiple predictors tends to cancel uncorrelated mistakes.

**Ensemble soft labels (what the student actually learns from)**
- We export teacher soft labels and train the student against these distributions.
- If we average probabilities, the ensemble distribution is:
$$p_{ens}(\cdot\mid x)=\frac{1}{M}\sum_{m=1}^M p_{t_m}(\cdot\mid x).$$
- Intuition: the student is trained to match a smoother and often better-calibrated target than any single teacher.

**Teacher validation results (from outputs/teachers/*/reliabilitymetrics.json)**

| Teacher | Raw Accuracy | Raw Macro-F1 |
|---|---:|---:|
| RN18 | 0.786182 | 0.780828 |
| B3 | 0.796091 | 0.790988 |
| CNXT | 0.794055 | 0.788959 |

**Note on ViT**
- Vision Transformers (ViT) are a strong teacher option, but a ViT teacher is not part of the current saved teacher artifacts in this repo yet.

**Speaker notes (English)**
- These are the three teacher architectures I currently have saved in the repo.
- They’re intentionally different: RN18 is a strong small CNN baseline, EfficientNet-B3 is a good accuracy–efficiency trade-off, and ConvNeXt has a more modern convolution design.
- The reason for multiple teachers is complementarity: they don’t make exactly the same errors.
- So the ensemble distribution becomes a smoother target for the student.
- This table is also a reality check: it shows what the teachers achieve before compression into a real-time student.

---

## Slide 9 / 19 — Losses Part 1 (CE and KD)

**Cross-Entropy (CE)**
- Standard supervised classification loss.
$$\mathcal{L}_{CE} = -\log p_s(y \mid x)$$

**CE is “hard supervision”**
- CE optimizes only the probability of the ground-truth class.
- It does not directly constrain *how probability mass is distributed among non-target classes*.

**Knowledge Distillation (KD)**
- Student matches teacher soft probabilities (temperature $T$) using KL divergence.
$$p_t^T(\cdot\mid x)=\mathrm{softmax}(z_t/T),\quad p_s^T(\cdot\mid x)=\mathrm{softmax}(z_s/T)$$
$$\mathcal{L}_{KD} = T^2\,\mathrm{KL}(p_t^T(\cdot\mid x)\ \|\ p_s^T(\cdot\mid x))$$
- Intuition: learns similarity between classes, not only the correct label.

**Common practical form (CE + KD)**
$$\mathcal{L}=(1-\alpha)\,\mathcal{L}_{CE}+\alpha\,\mathcal{L}_{KD}$$

**Decoupled KD (DKD)**
- Separates the distillation signal into target-class vs non-target-class parts.
- Intuition: can stabilize training and allow better weighting of “correct class” guidance vs “other classes” structure.

**Speaker notes (English)**
- Now I’ll explain the training objective.
- Cross-entropy is the standard supervised loss: it pushes the student to predict the ground-truth class.
- Knowledge distillation adds another term: match the teacher distribution using KL divergence at temperature $T$.
- So the student learns both the correct label and the teacher’s class-relationship structure.
- In practice we mix them with a weight $\alpha$ so we don’t drift away from ground truth.

---

## Slide 10 / 19 — Losses Part 2 (DKD: target vs non-target)

**DKD mechanism (compact)**
- Let $y$ be the ground-truth class.
- Define the *target vs non-target* 2-way distributions:
$$q_t=[p_t^T(y\mid x),\ 1-p_t^T(y\mid x)],\quad q_s=[p_s^T(y\mid x),\ 1-p_s^T(y\mid x)]$$
- Target-class KD (TCKD): distill how much mass the teacher assigns to the correct class vs all others.
$$\mathcal{L}_{TCKD}=\mathrm{KL}(q_t\ \|\ q_s)$$
- Non-target-class KD (NCKD): distill the relative structure among wrong classes conditioned on $\neg y$.
$$\mathcal{L}_{NCKD}=\mathrm{KL}(p_t^T(\cdot\mid x,\neg y)\ \|\ p_s^T(\cdot\mid x,\neg y))$$
- Combine:
$$\mathcal{L}_{DKD}=\alpha\,\mathcal{L}_{TCKD}+\beta\,\mathcal{L}_{NCKD}$$

**Why DKD can be more stable than KD (intuition)**
- Early in training, the student’s non-target distribution is noisy.
- Decoupling lets us transfer the correct-vs-rest signal without over-constraining the wrong-class structure too early.

**Speaker notes (English)**
- DKD is basically a refinement of KD.
- Instead of forcing the student to match the entire teacher distribution in one shot, it splits the signal into two parts.
- First: how much probability mass goes to the correct class versus the rest.
- Second: how the remaining probability mass is distributed among the wrong classes.
- Early on, the wrong-class structure can be noisy for the student, so this decoupling can be more stable.
- That’s why we do staged training: CE to stabilize, KD to transfer, and DKD to refine.

---

## Slide 11 / 19 — Calibration (NLL, ECE, Temperature Scaling)

**Why calibration matters**
- Real-time systems output confidence, not only labels.
- Overconfident wrong predictions are dangerous for user trust.

**Metrics we report**
- NLL: probabilistic correctness (lower is better).
- ECE: calibration gap between confidence and accuracy (lower is better).

**Temperature scaling (TS)**
- Post-hoc calibrates logits via a scalar $T_{cal}$:
$$p_{TS}(\cdot\mid x)=\mathrm{softmax}(z/T_{cal}).$$
- TS does not change argmax predictions; it mainly adjusts confidence.

**Speaker notes (English)**
- Since this is a real-time system, the confidence score matters.
- A model that is wrong with 99% confidence is much worse for user trust than a model that says “I’m not sure”.
- That’s why we report NLL and ECE, not only accuracy and F1.
- Temperature scaling is a simple post-hoc fix: it rescales logits and usually reduces overconfidence.
- So when you see “TS ECE” and “TS NLL” in the results, that’s calibration after temperature scaling.

---

## Slide 12 / 19 — Student Model Choice + Baseline Results (Real numbers)

**Student**
- MobileNetV3-Large (`mobilenetv3_large_100`): chosen for real-time efficiency.

**Why MobileNetV3 for real-time FER (theory)**
- Uses depthwise separable convolutions and inverted residual blocks to reduce FLOPs while preserving representational power.
- Good latency–accuracy trade-off on CPU/edge devices, which matches the real-time constraint.
- For FER, convolutional inductive bias (local texture + compositional features) is often strong under limited compute.

**Student baseline results (validation; repo artifacts)**

| Training stage | Epochs | Raw Acc | Raw Macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
|---|---:|---:|---:|---:|---:|---:|
| CE | 20 | 0.750174 | 0.741952 | 0.049897 | 0.777757 | 0.666403 |
| KD baseline | 5 | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| DKD baseline | 10 | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 |

**Interpretation**
- KD improves calibration after temperature scaling (lower TS ECE), but accuracy/F1 may not always increase in a short budget.
- DKD recovers/boosts performance compared to KD baseline in this setup.

**Important caveat about short-run budgets**
- In 5–10 epoch screenings, variance is higher and convergence is incomplete.
- Therefore we treat these as *directional evidence* and validate hypotheses under harder tests (domain shift).

**Speaker notes (English)**
- This is the student we actually deploy: MobileNetV3-Large.
- It’s chosen because it gives good accuracy per FLOP, which is exactly what we need for a real-time demo.
- This table is the baseline reference for the entire project: CE, then KD, then DKD.
- These runs are short-budget screenings, so I interpret them as trends.
- The key takeaway is that DKD is stronger than KD in this setup, and we always track calibration as well.

---

## Slide 13 / 19 — NegL (Theory: why it could help)

**Goal (why add-on methods exist)**
- Improve robustness and especially **minority-F1** without harming a strong DKD baseline.

**Core idea**
- NegL introduces a complementary supervision signal: instead of only “this is class $y$”, we sometimes add “this is *not* class $k$”.
- It is typically gated by uncertainty (e.g., entropy) so it acts mainly when the model is unsure.

**Why NegL might help minority classes (deeper framing)**
- Minority classes often suffer higher false positives/false negatives under imbalance and domain shift.
- NegL can reduce *overconfident confusion* by discouraging probability mass on implausible classes.
- In probabilistic terms: it pushes down wrong-class probabilities, which can improve calibration and reduce systematic confusions.

**Main failure mode**
- If the gate is wrong (applying NegL when the model is actually confident and correct), it can suppress true positives.

**Speaker notes (English)**
- Now I’ll talk about the two add-on ideas: NegL and NL.
- NegL is motivated by uncertainty: when the model is unsure, it can be safer to push down obviously wrong classes than to push up a possibly-wrong positive label.
- So NegL adds “negative evidence”, usually gated by entropy.
- The catch is that gating must be correct—otherwise we suppress true positives.
- Next, NL(proto) targets the feature space, not the logits.

---

## Slide 14 / 19 — NL (proto) (Theory: prototype memory + gating)

**Core idea**
- Maintain a prototype (moving-average centroid) per class in feature space.
- Add an auxiliary consistency/attraction term so features move toward their class prototype.

**Why it might help (deeper intuition)**
- Acts like a metric-learning regularizer: tighter intra-class clusters and larger inter-class separation.
- Can improve robustness if the feature space becomes more stable under nuisance factors (lighting/pose).

**Why it can fail (important)**
- Early training features can be unstable → prototypes become noisy.
- A strong auxiliary loss can then “pull” samples toward wrong prototypes, causing regression.
- Therefore the gate/weight schedule is not a detail; it is the method.

**Speaker notes (English)**
- NL(proto) is a prototype-based regularizer.
- The idea is: keep a moving-average prototype for each class in feature space, and encourage samples to be close to the prototype of their class.
- If it works, it makes representations more stable and can improve robustness under nuisance factors like lighting and pose.
- But it can fail if prototypes are noisy early on.
- So regression usually means the auxiliary loss is too strong or applied too early.

---

## Slide 15 / 19 — NL/NegL Ablations (Real numbers; concise)

**KD (5 epochs screening)**

| Run | Minority-F1 | Note |
|---|---:|---|
| KD baseline | 0.697342 | reference |
| KD + NegL (entropy=0.3, w=0.05, ratio=0.5) | 0.698288 | very small change; TS got worse |

**DKD (10 epochs screening)**

| Run | Minority-F1 | Note |
|---|---:|---|
| DKD baseline | 0.704458 | reference |
| DKD + NegL (entropy=0.3, w=0.05, ratio=0.5) | 0.705310 | tiny increase; TS ECE/NLL worse |
| DKD + NL(proto, top-k gate, w=0.1) | 0.688264 | clear regression |
| DKD + NegL + NL(proto) | 0.701544 | raw ECE/NLL improved; minority-F1 not improved |

**Speaker notes (English)**
- Now the key point: what happened in the real runs.
- These numbers come from our repo’s comparison tooling, so they’re directly auditable.
- NegL gives a tiny minority-F1 gain in the DKD screening at this setting, but it worsened calibration after temperature scaling.
- NL(proto) clearly regressed under the tested hyperparameters, which matches the prototype-instability failure mode.
- So at this stage, the honest conclusion is: we need a harder test—domain shift—before claiming robustness improvements.

---

## Slide 16 / 19 — What the Results Actually Say (Evidence-based)

**1) Baseline strength is real**
- DKD baseline is already strong on in-domain validation (Macro-F1 0.736796, Minority-F1 0.704458).

**2) NegL shows weak/partial evidence for minority-F1**
- Tiny minority-F1 improvement at entropy thresh=0.3 under DKD.
- But calibration after temperature scaling worsened in those runs.

**3) NL(proto) currently hurts DKD under the tested hyperparameters**
- This suggests either the auxiliary signal is too strong / misaligned, or the gate/weight needs redesign.

**Implication**
- To claim robustness improvements, we should test where domain mismatch is larger.

**Methodology note (why we don’t over-claim)**
- When deltas are small, they can be within run-to-run noise.
- Stronger evidence requires: (1) domain-shift evaluation, and ideally (2) repeated seeds or longer budgets.

**Speaker notes (English)**
- Let me summarize the evidence in plain terms.
- The DKD baseline is already strong in-domain, so big gains are unlikely in a short budget.
- NegL is not a consistent win yet—it shows only weak evidence and introduces calibration trade-offs.
- NL(proto), as tested, is clearly unsafe because it regressed.
- So the correct research move is not more storytelling—it’s better evidence: evaluate under domain shift where robustness differences should show up.

---

## Slide 17 / 19 — Domain Shift Evaluation Plan (Minority-F1 is the primary metric)

**Why domain shift tests are necessary**
- In-domain metrics can saturate; robustness gains often appear under distribution shift.

**What “domain shift” means here (theory)**
- **Covariate shift**: $p(x)$ changes (lighting/pose/camera), while the label space stays the same.
- FER is sensitive to subtle textures; shift in resolution, blur, or illumination can disproportionately hurt minority expressions.

**Why minority-F1 is a good primary metric**
- Accuracy can hide failure on rare but important classes.
- Minority-F1 (lowest-3) is a simple *worst-group robustness proxy*: it asks whether the system is usable on the hardest/rarest expressions.
- It aligns with real deployment risk: consistent failure on Fear/Disgust is unacceptable even if Neutral/Happy dominate.

**Concrete evaluation target**
- Evaluate saved student checkpoints on ExpW test split.
- Primary metric: **Minority-F1 (lowest-3)**.

**One-click evaluation (implemented in repo)**
- `scripts/run_domain_shift_eval_oneclick.ps1`
- Example:
	- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_domain_shift_eval_oneclick.ps1 -EvalManifest Training_data_cleaned/expw_full_manifest.csv -EvalSplit test`

**Outputs**
- `outputs/evals/students/<run>__expw_full_manifest__test__<stamp>/reliabilitymetrics.json`
- `outputs/evals/_compare_<stamp>_domainshift_expw_full_manifest_test.md`

**Speaker notes (English)**
- This is the next experiment that can actually validate the hypothesis.
- We take the saved student checkpoints and evaluate on ExpW test.
- The primary metric is minority-F1, because it captures worst-group behavior and is sensitive to robustness failures.
- The good news is the tooling is already implemented: one command generates evaluation outputs and a comparison markdown table.
- So the results will be reproducible and easy to review.

---

## Slide 18 / 19 — Real-time Deployment Architecture (System view)

**Runtime pipeline**
1) Face detection (YuNet / ONNX)
2) Preprocessing (CLAHE for lighting normalization)
3) Expression classification (MobileNetV3 student)
4) Output: label + calibrated confidence

**Why this structure**
- Separates detection from classification (modularity and speed).
- Preprocessing reduces sensitivity to illumination changes (domain shift mitigation).

**Speaker notes (English)**
- Finally, this is how the system runs in real time.
- We do face detection first, then preprocessing, then the student classifier.
- The reason for this modular design is engineering stability: improvements in detection don’t require retraining the classifier.
- CLAHE is a practical step to reduce lighting sensitivity, which is a major domain-shift factor for webcams.
- Teachers and ensembles are purely offline; only the student needs to meet latency constraints.

---

## Slide 19 / 19 — Conclusion and Next Steps

**Conclusion**
- Built a reproducible teacher → soft label → student pipeline suitable for a real-time FER system.
- Established strong baselines (CE, KD, DKD) with real metrics saved in repo artifacts.
- Early NL/NegL screenings show: NegL may slightly help minority-F1 but is not a consistent win; NL(proto) needs redesign to avoid regressions.

**Next steps (professor-aligned)**
- Run domain-shift evaluation (ExpW test) and judge improvements primarily by minority-F1.
- If NegL is promising, do a small safe hyperparameter sweep (lower weight) under domain shift.
- If NL is revisited: reduce weight, change gating schedule, and validate that it does not regress DKD.
- Real-time stability: add temporal smoothing / hysteresis to reduce prediction flicker.

**Speaker notes (English)**
- To conclude: the main contribution is an end-to-end, reproducible pipeline plus a working real-time demo.
- We established strong baselines with CE, KD, and DKD, and all the metrics are saved as repo artifacts.
- The add-on methods, NegL and NL, are not consistent improvements yet—especially NL, which regressed under current settings.
- So the next step is simple and scientific: run domain-shift evaluation on ExpW test, focusing on minority-F1.
- Then we can make an informed choice: either carefully tune NegL for robustness, or redesign NL so it’s stable.

---

## Oral Defense Q&A (appendix; not counted as slides)

**Q1: Why use teacher–student distillation instead of just training the student with CE longer?**
- Distillation provides richer targets (soft class structure), which can improve generalization and calibration under limited student capacity.
- It also leverages stronger offline compute during teacher training while keeping the deployed model lightweight.

**Q2: Why DKD instead of standard KD?**
- KD forces the student to match the full teacher distribution, including noisy non-target structure early in training.
- DKD decouples target-vs-rest from non-target structure, which can be more stable and tunable via $\alpha,\beta$.

**Q3: Why is minority-F1 prioritized over accuracy?**
- In imbalanced multi-class problems, accuracy can stay high while rare classes fail completely.
- Minority-F1 (lowest-3) is a simple robustness proxy: it measures whether the model is acceptable on the hardest classes.

**Q4: Why do you claim domain shift matters here?**
- FER is sensitive to illumination/pose/blur; webcam data differs from curated datasets.
- Robustness improvements often appear under shift even when in-domain metrics are saturated.

**Q5: Why did NL(proto) regress—does that invalidate the idea?**
- It shows the current hyperparameters/gating are not safe.
- Prototype-based losses are sensitive to early representation instability; a redesigned schedule or weaker weighting may be required.

**Q6: If NegL slightly improves minority-F1 but worsens calibration after TS, what do you do next?**
- Verify under domain shift first; then adjust NegL weight/gate to trade off minority-F1 vs calibration.
- If the goal is deployment, calibration and stability constraints can be treated as acceptance criteria.

