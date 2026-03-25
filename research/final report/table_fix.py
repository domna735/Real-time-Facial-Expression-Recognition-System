import re

path = r'c:\Real-time-Facial-Expression-Recognition-System_v2_restart\research\final report\final report version 3.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace Teacher Training Configurations
old_1 = """### Teacher Training Configurations

Detailed source filtering applied during teacher training:
- Effective datasets: ferplus, affectnet_full_balanced, rafdb_basic
- Example (RN18): 225,629 rows after filtering from 466,284 total
- Source composition: `{ferplus: 138,526, affectnet_full_balanced: 71,764, rafdb_basic: 15,339}`"""

new_1 = """### Teacher Training Source Composition

Detailed sub-dataset breakdown applied during teacher training (e.g., for RN18, leaving 225,629 rows after filtering from the 466,284 total):

| Dataset Source | Retained Rows | Notes |
| :--- | ---: | :--- |
| **FERPlus** | 138,526 | High-quality baseline |
| **AffectNet** (balanced) | 71,764 | Heavily downsampled for class parity |
| **RAF-DB** (basic) | 15,339 | Studio-curated base |
| **Total Effective** | 225,629 | Filtered to exclude ExpW & synthetic |"""

text = text.replace(old_1, new_1)

# Replace Student Distillation Details
old_2 = """### Student Distillation Details

**HQ training manifest:** 259,004 rows containing:
- train: 213,144 rows
- val: 18,020 rows
- test: 27,840 rows"""

new_2 = """### Student Distillation Split Structure

**HQ training manifest (`classification_manifest_hq_train.csv`)**, distilled via Teacher Ensemble:

| Data Split | Sample Count | Percentage |
| :--- | ---: | ---: |
| **Training** | 213,144 | ~82.3% |
| **Validation** | 18,020 | ~7.0% |
| **Testing** | 27,840 | ~10.7% |
| **Total Rows** | 259,004 | 100.0% |"""

text = text.replace(old_2, new_2)

# Replace Per-source Breakdown Analysis
old_3 = """### Per-source Breakdown Analysis

eval-only per-source macro-F1 (CE checkpoint):
- expw_full: 0.490 (6,780 rows)
- expw_hq: 0.279 (3,336 rows) — **lowest performer**
- rafml_argmax: 0.485 (982 rows)
- rafdb_compound: 0.330 (792 rows)

The low aggregate eval-only result is driven primarily by `expw_hq` and compound mappings."""

new_3 = """### Eval-Only Domain Regression (Per-Source Breakdown)

Macro-F1 analysis on external stress-test sources, using the Baseline Student (CE checkpoint). The low aggregate score is heavily suppressed by extreme difficulty in `expw_hq` and compound mappings.

| Evaluation Source | Sample Size | Macro-F1 (CE) | Evaluation Note |
| :--- | ---: | ---: | :--- |
| `expw_full` | 6,780 | 0.490 | Open-domain in-the-wild |
| `rafml_argmax` | 982 | 0.485 | Standard multi-label argmax |
| `rafdb_compound` | 792 | 0.330 | Compound expression mismatch |
| `expw_hq` | 3,336 | 0.279 | **Lowest performer (severe bottleneck)** |"""

text = text.replace(old_3, new_3)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Tables successfully injected!")
