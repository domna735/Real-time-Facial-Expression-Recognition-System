# Suggestion: data cut down (keep higher image size)

## Goal
You want to keep higher training resolution (e.g., 384×384) but reduce wall-clock time by cutting the 466,284-row unified manifest (`Training_data_cleaned/classification_manifest.csv`) down to a smaller, higher-quality subset.

Key point: at 384×384, total images is the biggest time driver. Reducing validation/export frequency helps, but dataset size dominates.

---

## What “better quality data” means (practical definition)
For this project, “better quality” usually means:
- Face is clearly visible and relatively large in the image.
- Labels are reliable (human-annotated or well-curated).
- Images are already aligned/cropped (less background / less ambiguity).
- Less domain noise (random web images often contain label noise).

Note: “HQ” can mean two different things:
- **Label-HQ** (more reliable labels): `ferplus`, `rafdb_basic`.
- **Image-HQ** (clear cropped faces): `expw_hq` (and RAF aligned images).
`ferplus` images are originally 48×48; at 384×384 they are upsampled (label-HQ, not pixel-HQ).

---

## FERPlus 48×48 issue (what to do)
Training at 384×384 with 48×48 sources can waste compute and may add “pixel-noise” patterns (because the model sees very little real detail).

Recommended options (simple → stronger):

### Option 1 (recommended): two-stage schedule
- **Stage A (fast):** train using the curated set **including** `ferplus` at **224×224** (or 256).
- **Stage B (HQ finetune):** finetune at **384×384** on an HQ subset **excluding `ferplus`** (keep `rafdb_basic`, `affectnet_full_balanced`, `expw_hq`).

Why: you still benefit from FERPlus label signal, but you don’t pay 384×384 compute on 48×48 pixels.

### Option 2: keep FERPlus only as evaluation
- Remove `ferplus` from the 384×384 teacher training manifest.
- Keep `ferplus` test split as a clean benchmark in evaluation.

### Option 3: keep FERPlus in training but reduce its impact
- Keep it in training (for label-HQ) but **cap** it or downsample it.
- This reduces how often the model sees upsampled 48×48 inputs.

Not recommended (high cost / uncertain benefit):
- Super-resolution preprocessing for FERPlus (adds complexity and can hallucinate details).

---

## Recommended approach (best trade-off)
### Recommendation A (HQ subset for teacher training at 384)
Train teachers on a **curated multi-dataset subset**:
- Keep (high value / typically cleaner):
  - `ferplus` (cleaner labels; strong baseline dataset)
  - `rafdb_basic` (aligned faces; reliable labels)
  - `affectnet_full_balanced` (large and diverse; keep but optionally cap it)
  - `expw_hq` (cropped faces + confidence-filtered; good “in-the-wild” supplement)
- Drop or deprioritize (more label uncertainty / mapping artifacts):
  - `rafdb_compound_mapped` (compound→base mapping adds ambiguity)
  - `rafml_argmax` (argmax from multi-label distribution can be noisy)
  - `affectnet_yolo_format` (keep only if you specifically need it; otherwise it can be redundant)
- Use `expw_full` mainly as **robustness evaluation** (noisy web labels).

Why this is good:
- You keep diverse domains (lab + curated + wild HQ) but remove the noisiest sources.
- You still get “in-the-wild” exposure via `expw_hq`, without paying the cost of all `expw_full`.

---

## What we generated (CSV manifests you can use now)
To make this actionable immediately, we generated two new manifests:

**1) Curated training manifest (HQ training set)**
- File: `Training_data_cleaned/classification_manifest_hq_train.csv`
- Sources included:
  - `ferplus`
  - `rafdb_basic`
  - `affectnet_full_balanced`
  - `expw_hq` (appended from `Training_data_cleaned/expw_hq_manifest.csv`)
- Rows: **259,004**
  - `ferplus`: 138,526
  - `affectnet_full_balanced`: 71,764
  - `expw_hq`: 33,375
  - `rafdb_basic`: 15,339

**2) Evaluation-only manifest (datasets we do NOT train on)**
- File: `Training_data_cleaned/classification_manifest_eval_only.csv`
- Sources included:
  - `expw_full` (robustness / noisy-in-the-wild)
  - `rafml_argmax` (noisy label via argmax)
  - `rafdb_compound_mapped` (compound→base mapping)
  - `expw_hq` (optional extra evaluation)
  - Rows: **134,030** (before de-dup)
    - `expw_full`: 91,793
  - `expw_hq`: 33,375
  - `rafml_argmax`: 4,908
  - `rafdb_compound_mapped`: 3,954

  Optional de-duplication (recommended):
  - We added `--dedupe` to drop repeated rows by exact `image_path`.
  - After de-dup, eval-only rows become **110,333**, and `expw_full` becomes **68,096** because many `expw_full` image paths overlap with the HQ-exported crops/paths already present elsewhere.

Tool used to generate them:
- `tools/data/build_curated_manifests.py`

---

## Decisions locked (current plan)
- **Training manifest:** `Training_data_cleaned/classification_manifest_hq_train.csv` (259,004 rows).
- **Evaluation-only manifest:** `Training_data_cleaned/classification_manifest_eval_only.csv` (use the de-dup version; total 110,333 rows; `expw_full` 68,096).
- **Evaluation cadence:** run validation/testing only every **10 epochs** (`--eval-every 10`).
- **Resume behavior:** unchanged; resume always uses `checkpoint_last.pt` in the run `--output-dir`.
- **Duplicates:** de-dup enabled via `tools/data/build_curated_manifests.py --dedupe`.

---

## Should we split training vs testing CSV?
Yes (recommended), but with one important note:
- Splitting train vs test CSVs improves *clarity* and prevents accidental training on noisy sources.
- It only improves *runtime* if your evaluation loop is expensive (large `val/test`) and you run it too often.

So, the best time-saving combo is:
- Train on the curated training manifest.
- Validate less frequently: set `--eval-every 10` during teacher training.
- Run the evaluation-only manifest only at milestones (e.g., epoch 10/30/60) or after training.

Resume + checkpoints:
- Your stop/resume workflow is unaffected by this change. Resume still comes from `checkpoint_last.pt` in the run `--output-dir`.
- Testing frequency does not affect writing `checkpoint_last.pt` (it is saved every epoch).

---

## What is best for training vs best for testing
### Best for training (clean + stable + still diverse)
- `ferplus` + `rafdb_basic` are the most label-reliable “core”.
- `affectnet_full_balanced` adds diversity/scale (but is a major runtime driver).
- `expw_hq` adds in-the-wild variation with a quality filter.

### Best for testing (to measure generalization / robustness)
- **Primary robustness test:** `expw_full` (noisy, hard; good for “real-world” stress testing)
- **Clean benchmarks:** `rafdb_basic` test split, `ferplus` test split
- **HQ wild benchmark:** `expw_hq` test split

---

## Suggested dataset size (based on your prior results)
In your interim report, the consolidated dataset used was **228,615** samples after applying quality filters and caps.

For 384×384 teacher training now, a practical target is:
- **Fast iteration:** 120k–180k
- **Recommended:** 200k–280k (your current curated train = **259k**, which fits here)
- **Large:** 300k+ (expect long epochs at 384×384)

If runtime is still too slow with 259k at 384×384, the first cut should usually be:
- cap `affectnet_full_balanced` (it is the biggest component in the curated set)


---

## If you want to cut even more (fastest options)
### Recommendation B (single-core dataset for training)
Yes, you *can* train using only one dataset.

If you pick only one dataset for training, the most reasonable choice is:
- **Train:** `affectnet_full_balanced`
- **Validate/Test (in-domain):** AffectNet’s own splits
- **Test (out-of-domain):** `expw_hq` and/or `expw_full`

Trade-offs:
- Pros: simplest, stable, and typically strong.
- Cons: bigger domain shift when testing on RAF/FERPlus/ExpW; you may underperform on “lab-style” aligned faces.

### Recommendation C (two-stage: fast pretrain + high-res finetune)
If time is very tight:
1) Pretrain at **224×224** on a larger set (fast), then
2) Finetune at **384×384** on a smaller HQ set.

This often preserves accuracy while dramatically reducing total 384×384 compute.

---

## Concrete “cut-down recipes” (easy experiments)
### Recipe 1: HQ-Only (small, fastest at 384)
- Use only: `rafdb_basic` + `ferplus` + `expw_hq`
- Expected size: relatively small, good label quality.
- Use case: quickest teacher iteration to validate pipeline / distillation.

### Recipe 2: HQ + Capped AffectNet (medium, recommended)
- Use: Recipe 1 + `affectnet_full_balanced` but **cap** per class (example: max N per class per split)
- Use case: keep diversity without letting AffectNet dominate runtime.

### Recipe 3: Full minus noisy (large)
- Use: everything except `rafml_argmax` + `rafdb_compound_mapped` + maybe `expw_full`.
- Use case: if you still want a big dataset but slightly cleaner.

---

## How to cut the data (implementation options)
### Option 1: Build a new “curated” manifest (recommended)
Create a new CSV (example name):
- `Training_data_cleaned/classification_manifest_curated.csv`

Rules to implement in the manifest builder:
- Filter by `source` (keep/drop sources listed above).
- Optional: cap per-source and/or per-class to control size.
- Keep splits stable (`train/val/test`) so evaluation is comparable.

### Option 2: Filter `expw_full` using quality signals you already have
`expw_full_manifest.csv` already contains:
- `confidence` (bbox confidence)
- `bbox_*` (face box)

Suggested filters:
- `confidence >= 70` (or 80 if you want very strict)
- minimum bbox size (e.g., width and height >= 80 px)

This is the cleanest “quality filter” available right now without running extra image analysis.

### Option 3: Add image-based quality filters (strong but takes time to implement)
If you want true “image quality” filtering:
- minimum resolution threshold
- blur filter (Laplacian variance)
- optional face detector check (face present, face size ratio)

This requires scanning images once and writing a filtered manifest.

---

## My recommendation (what to keep right now)
If the main goal is to finish teacher training at 384×384 in reasonable time:
- **Keep for teacher training:** `ferplus`, `rafdb_basic`, `expw_hq`, and a **capped** `affectnet_full_balanced`.
- **Use for testing/robustness:** `expw_full` (and optionally RAF subsets as extra tests).
- **Drop initially:** `rafml_argmax`, `rafdb_compound_mapped`.

---

## Next step (if you want me to implement it)
I can add a small tool that generates `classification_manifest_curated.csv` from your existing manifests with:
- keep/drop sources
- optional caps per source/class
- optional ExpW confidence/bbox filters

Tell me your target size (e.g., 100k / 150k / 200k rows) and whether ExpW should be:
- training: `expw_hq` only
- testing: `expw_full` only
- or both
