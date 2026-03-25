
# Error & Improvement Checklist (Repo-wide)

Document purpose: capture **real errors** (things that break execution or correctness) and **high-value improvements** (things that reduce future bugs / improve reproducibility), based on an actual sanity pass over this workspace.

Last updated: 2026-02-05

---

## 0) What was checked (so results are interpretable)

### Automated sanity checks executed

- Python syntax/bytecode compile (selected code dirs):
	- Command (Windows venv): `./.venv/Scripts/python.exe -m compileall -q src scripts tools demo "project with Liang.Nicholas"`
- Dependency integrity check:
	- Command: `./.venv/Scripts/python.exe -m pip check`
- Final report audit script:
	- Command: `./.venv/Scripts/python.exe scripts/audit_final_report.py`
- Final report artifact-path existence check:
	- Verified that **all repo-relative backticked paths** in `research/final report/final report.md` exist (excluding globs like `outputs/students/**/reliabilitymetrics.json`).

### Editor diagnostics collected

- VS Code “Problems” panel reported many Markdown lint items (mostly formatting rules like `MD025`, `MD010`, `MD007`, `MD060`).
	- These are **not runtime failures**, but they *will* show as “errors” if you run markdownlint.

---

## 1) Critical / real errors (break code execution)

### 1.1 Fixed: `scripts/realtime_infer_arcface.py` had a syntax error

- **Severity:** FAIL (this file could not run)
- **Symptom:** `IndentationError: unexpected indent` around the `--device` argument definition.
- **Root cause:** a block of code was accidentally indented under another `ap.add_argument(...)` call, and a corrupted `if not demo_path.exists():` block contained stray tokens.
- **Fix applied (2026-02-05):**
	- Corrected indentation.
	- Repaired the missing-demo-script guard.
	- Ensured `--device` is forwarded into `demo/realtime_demo.py`.

✅ Post-fix validation:
- `compileall` exit code `0`.
- Import check of `demo/realtime_demo.py` succeeds.

---

## 2) Correctness / reproducibility checks (report + artifacts)

### 2.1 Final report numeric consistency audit

- **Status:** OK
- Script: `scripts/audit_final_report.py`
- Result: `OK: 13 | WARN: 0 | FAIL: 0` and “No numeric mismatches found for audited tables.”

This is strong evidence that the core reported tables match the JSON artifacts you cite.

### 2.2 Final report artifact-path existence

- **Status:** OK
- `research/final report/final report.md` contains many backticked repo-relative artifact paths.
- Spot-check + automated check confirms: **no missing referenced artifact paths** (excluding intended wildcards / globs).

---

## 3) Non-critical but important issues (won’t crash, but will confuse / reduce quality)

### 3.1 Markdown lint “errors” in `final report.md` are mostly style-rule conflicts

- File: `research/final report/final report.md`
- Observed rule: `MD025/single-title/single-h1` (multiple `#` headings)

This is not a content mistake: a long report often uses multiple top-level headings. The “error” is just markdownlint configuration.

**Recommended improvement (pick one):**

1) **Prefer (low effort):** add a repo markdownlint config to disable MD025 (and optionally MD010/MD007/MD060 if you don’t care about them).
2) Convert the report to a single H1 and make sections H2 (`##`) / H3 (`###`).

If you want a clean “Problems” panel, option (1) is usually best.

### 3.2 Other markdownlint issues in research notes (tabs / list indentation)

- Example file: `research/FYP Paper/Paper study report.md`
- Issues seen in VS Code:
	- `MD010/no-hard-tabs` (hard tabs)
	- `MD007/ul-indent` (list indentation)
	- `MD025` (multiple H1 headings)

These are presentation/style issues, not research-content correctness issues.

---

## 4) Environment / dependency improvements (reduce “works on my machine” risk)

### 4.1 Windows: `python` command may resolve to Microsoft Store alias

- Observed symptom during checks: `python` was not found unless using `./.venv/Scripts/python.exe`.

**Recommended improvement:**
- In docs/commands, always show venv-explicit commands on Windows, e.g.
	- `./.venv/Scripts/python.exe scripts/train_student.py ...`
- Optional: add a short troubleshooting note (“disable App Execution Alias for python.exe”).

### 4.2 `requirements.txt` vs `requirements-directml.txt` diverge (expected), but pinning strategy can be clearer

- `requirements.txt` pins a CUDA nightly build line:
	- `torch==2.11.0.dev20251215+cu128` (+ matching nightly torchvision/torchaudio)
- `requirements-directml.txt` pins a stable-ish DML stack:
	- `torch==2.4.1`, `torch-directml==0.2.5.dev240914`, `torchvision==0.19.1`

**Risk:** nightly CUDA packages reduce reproducibility across machines and time.

**Recommended improvement:**
- Consider adding a third file like `requirements-cuda-stable.txt` (stable PyTorch release) if you want reproducible CUDA.
- Consider recording the exact wheel index / install commands used for CUDA builds (PyTorch’s extra-index URL, etc.).
- (Optional) move to a lockfile approach (`pip-tools` / `uv lock`) if you want full reproducibility.

---

## 5) Repo structure / maintainability improvements

### 5.1 Duplicate code tree: `github_demo_repo/` mirrors the main repo

- This workspace includes both:
	- top-level code (`src/`, `demo/`, `scripts/`, `tools/`)
	- and a near-duplicate under `github_demo_repo/`

**Risk:** easy to fix a bug in one copy but not the other.

**Recommended improvement (choose one):**

1) Treat `github_demo_repo/` as an exported snapshot only (never edit), and add a short note in its README.
2) Remove duplication and generate the demo repo via a script (copy/export step).
3) Convert it into a git submodule/subtree or a packaging layout so there is only one “source of truth.”

### 5.2 Add a minimal CI “sanity gate” for future regressions

Even without full training tests, you can prevent obvious breakage with:

- `python -m compileall -q src scripts tools demo`
- `python -m pip check`
- `python scripts/audit_final_report.py`

This would have caught the `scripts/realtime_infer_arcface.py` indentation break immediately.

### 5.3 (Optional) Enforce formatting / linting for Python

- Add `ruff` + `ruff format` (or black) to standardize style.
- Add a pre-commit hook to catch syntax/lint issues before commits.

---

## 6) Nice-to-have report improvements (not required, but strengthen the paper)

### 6.1 Make “artifact-backed” claims even more audit-friendly

The report is already strong here. Two small tweaks that can help readers:

- For key metrics tables, add a one-line “Audit: scripts/audit_final_report.py (PASS)” note.
- For any wildcard references like `outputs/students/**/reliabilitymetrics.json`, consider adding one concrete example path in addition to the wildcard.

---

## 7) Next actions (priority order)

1) ✅ Done: keep `scripts/realtime_infer_arcface.py` fixed and re-run `compileall` after changes.
2) Decide how you want to handle markdownlint rules (`MD025` etc.): configure markdownlint vs rewrite headings.
3) Add a tiny CI/sanity script that runs `compileall` + `pip check` + `audit_final_report.py`.
4) Decide the “single source of truth” approach for `github_demo_repo/`.

