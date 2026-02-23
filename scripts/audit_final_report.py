import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "research" / "final report" / "final report.md"


def _read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> Any:
	return json.loads(_read_text(path))


def _to_float(s: str) -> Optional[float]:
	s = s.strip()
	if s in {"-", "", "—"}:
		return None
	try:
		return float(s)
	except ValueError:
		return None


@dataclass
class Finding:
	level: str  # OK/WARN/FAIL
	where: str
	message: str


def parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[List[str], List[Dict[str, str]], int]:
	"""Parse a GitHub-flavored markdown table starting at start_idx.

	Returns: (headers, rows, next_index_after_table)
	"""

	header_line = lines[start_idx]
	if "|" not in header_line:
		raise ValueError("Not a table header")

	headers = [h.strip() for h in header_line.strip().strip("|").split("|")]

	# separator line should exist
	sep_idx = start_idx + 1
	if sep_idx >= len(lines):
		raise ValueError("Missing table separator")

	rows: List[Dict[str, str]] = []
	i = start_idx + 2
	while i < len(lines):
		line = lines[i]
		if not line.strip():
			break
		if not line.lstrip().startswith("|"):
			break
		cells = [c.strip() for c in line.strip().strip("|").split("|")]
		row: Dict[str, str] = {}
		for j, h in enumerate(headers):
			row[h] = cells[j] if j < len(cells) else ""
		rows.append(row)
		i += 1

	return headers, rows, i


def find_table_by_header(lines: List[str], header_prefix: str) -> Tuple[int, List[str], List[Dict[str, str]]]:
	for idx, line in enumerate(lines):
		if line.strip().startswith(header_prefix):
			headers, rows, _ = parse_markdown_table(lines, idx)
			return idx, headers, rows
	raise FileNotFoundError(f"Table with header prefix not found: {header_prefix}")


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
	return abs(a - b) <= tol


def report_matches_artifact(report_str: str, truth: float) -> bool:
	"""Return True if a numeric cell in the report is consistent with the artifact value.

	The report often rounds to a fixed number of decimals (e.g., 4 d.p.). We accept a
	value if it matches the artifact within half a unit of the reported precision.
	"""

	report_str = report_str.strip()
	got = _to_float(report_str)
	if got is None:
		return False

	# scientific notation or non-simple numeric: fall back to small epsilon
	if "e" in report_str.lower():
		return approx_equal(got, truth, tol=1e-6)

	m = re.match(r"^-?\d+(?:\.(\d+))?$", report_str)
	if not m:
		return approx_equal(got, truth, tol=1e-6)

	decimals = len(m.group(1) or "")
	if decimals == 0:
		return abs(got - truth) <= 0.5

	unit = 0.5 * (10 ** (-decimals))
	return abs(got - truth) <= unit + 1e-12


def check_exists(path_str: str) -> bool:
	# supports simple globs
	if any(ch in path_str for ch in "*?["):
		return len(list(REPO_ROOT.glob(path_str.replace("\\", "/")))) > 0
	return (REPO_ROOT / path_str).exists()


def audit_paths(report_text: str) -> List[Finding]:
	findings: List[Finding] = []
	paths = re.findall(r"`([^`]+)`", report_text)

	candidate_paths = []
	for p in paths:
		if "/" in p or "\\" in p:
			candidate_paths.append(p)

	checked = set(candidate_paths)
	missing: List[str] = []
	for p in sorted(checked):
		p_norm = p.replace("\\", "/")
		if re.match(r"^[A-Za-z]:/", p_norm):
			findings.append(Finding("WARN", "artifact path", f"Absolute path in backticks (prefer repo-relative): {p}"))
			continue
		if not check_exists(p_norm):
			missing.append(p)

	if missing:
		for p in missing[:50]:
			findings.append(Finding("FAIL", "artifact path", f"Referenced path not found: {p}"))
	else:
		findings.append(Finding("OK", "artifact path", "All backticked repo-relative paths appear to exist (or are globs with matches)."))

	return findings


def audit_dataset_counts(report_text: str) -> List[Finding]:
	findings: List[Finding] = []

	manifest_validation = _load_json(REPO_ROOT / "outputs" / "manifest_validation_all_with_expw.json")
	rows_total = int(manifest_validation.get("rows_total"))
	missing_paths = int(manifest_validation.get("missing_paths"))
	bad_labels = int(manifest_validation.get("bad_labels"))

	def _extract_int(pattern: str, label: str) -> Optional[int]:
		m = re.search(pattern, report_text)
		if not m:
			findings.append(Finding("FAIL", "4.1 Dataset integrity", f"Could not find '{label}' line in report."))
			return None
		return int(m.group(1).replace(",", ""))

	rep_rows = _extract_int(r"Total rows:\s*([0-9,]+)", "Total rows")
	rep_missing = _extract_int(r"Missing paths:\s*([0-9,]+)", "Missing paths")
	rep_bad = _extract_int(r"Bad labels:\s*([0-9,]+)", "Bad labels")

	if rep_rows is not None and rep_rows != rows_total:
		findings.append(Finding("FAIL", "4.1 Dataset integrity", f"Total rows mismatch: report={rep_rows} artifact={rows_total}"))
	if rep_missing is not None and rep_missing != missing_paths:
		findings.append(Finding("FAIL", "4.1 Dataset integrity", f"Missing paths mismatch: report={rep_missing} artifact={missing_paths}"))
	if rep_bad is not None and rep_bad != bad_labels:
		findings.append(Finding("FAIL", "4.1 Dataset integrity", f"Bad labels mismatch: report={rep_bad} artifact={bad_labels}"))

	# split sizes come from outputs/manifest_counts_summary.json
	counts = _load_json(REPO_ROOT / "outputs" / "manifest_counts_summary.json")
	by_path: Dict[str, Any] = {Path(item["path"]).name: item for item in counts}

	def _check_split_line(filename: str) -> None:
		item = by_path.get(filename)
		if item is None:
			findings.append(Finding("FAIL", "4.1 Dataset integrity", f"manifest_counts_summary.json missing entry for {filename}"))
			return

		splits = item["splits"]
		# match: `.../file.csv` split sizes: train=... / val=... / test=...
		m = re.search(
			rf"`[^`]*{re.escape(filename)}` split sizes:\s*train=([0-9,]+)\s*/\s*val=([0-9,]+)\s*/\s*test=([0-9,]+)",
			report_text,
		)
		if not m:
			findings.append(Finding("FAIL", "4.1 Dataset integrity", f"Could not find split sizes line for {filename} in report."))
			return

		rep_train = int(m.group(1).replace(",", ""))
		rep_val = int(m.group(2).replace(",", ""))
		rep_test = int(m.group(3).replace(",", ""))

		if rep_train != int(splits["train"]):
			findings.append(
				Finding(
					"FAIL",
					"4.1 Dataset integrity",
					f"{filename} train split mismatch: report={rep_train} artifact={splits['train']}",
				)
			)
		if rep_val != int(splits["val"]):
			findings.append(
				Finding(
					"FAIL",
					"4.1 Dataset integrity",
					f"{filename} val split mismatch: report={rep_val} artifact={splits['val']}",
				)
			)
		if rep_test != int(splits["test"]):
			findings.append(
				Finding(
					"FAIL",
					"4.1 Dataset integrity",
					f"{filename} test split mismatch: report={rep_test} artifact={splits['test']}",
				)
			)

	_check_split_line("classification_manifest.csv")
	_check_split_line("classification_manifest_hq_train.csv")

	# mixed-source benchmark size (test_all_sources.csv)
	test_all_item = by_path.get("test_all_sources.csv")
	if test_all_item is not None:
		m = re.search(r"test_all_sources\.csv`[^\n]*?([0-9,]+) rows", report_text)
		if m:
			rep_total = int(m.group(1).replace(",", ""))
			if rep_total != int(test_all_item["total_rows"]):
				findings.append(
					Finding(
						"FAIL",
						"4.1 Dataset integrity",
						f"test_all_sources.csv total_rows mismatch: report={rep_total} artifact={test_all_item['total_rows']}",
					)
				)
		else:
			findings.append(Finding("WARN", "4.1 Dataset integrity", "Could not locate test_all_sources row-count line to check."))

	# eval-only size appears as: verified size: 110,333 rows
	eval_item = by_path.get("classification_manifest_eval_only.csv")
	if eval_item is not None:
		m = re.search(r"classification_manifest_eval_only\.csv` \(verified size:\s*([0-9,]+) rows\)", report_text)
		if m:
			rep_total = int(m.group(1).replace(",", ""))
			if rep_total != int(eval_item["total_rows"]):
				findings.append(
					Finding(
						"FAIL",
						"3.7 Domain shift track",
						f"Eval-only manifest total_rows mismatch: report={rep_total} artifact={eval_item['total_rows']}",
					)
				)
		else:
			findings.append(Finding("WARN", "3.7 Domain shift track", "Could not locate eval-only verified-size line to check."))

	if not any(f.level == "FAIL" and f.where in {"4.1 Dataset integrity", "3.7 Domain shift track"} for f in findings):
		findings.append(Finding("OK", "4.1 Dataset integrity", "Dataset integrity counts and key split sizes match artifacts."))

	return findings


def audit_abstract_teacher_range(report_text: str) -> List[Finding]:
	findings: List[Finding] = []

	# Parse: "macro-F1 ≈ 0.781–0.791" (en dash)
	m = re.search(r"macro-F1\s*≈\s*([0-9]+\.[0-9]+)\s*[–-]\s*([0-9]+\.[0-9]+)", report_text)
	if not m:
		findings.append(Finding("WARN", "Abstract", "Could not find teacher macro-F1 range statement in abstract."))
		return findings

	rep_lo = float(m.group(1))
	rep_hi = float(m.group(2))

	teacher_paths = [
		REPO_ROOT / "outputs" / "teachers" / "RN18_resnet18_seed1337_stageA_img224" / "reliabilitymetrics.json",
		REPO_ROOT / "outputs" / "teachers" / "B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224" / "reliabilitymetrics.json",
		REPO_ROOT / "outputs" / "teachers" / "CNXT_convnext_tiny_seed1337_stageA_img224" / "reliabilitymetrics.json",
	]
	vals = [float(_load_json(p)["raw"]["macro_f1"]) for p in teacher_paths]
	truth_lo = round(min(vals), 3)
	truth_hi = round(max(vals), 3)

	if rep_lo != truth_lo or rep_hi != truth_hi:
		findings.append(
			Finding(
				"FAIL",
				"Abstract",
				f"Teacher macro-F1 range mismatch: report≈{rep_lo}–{rep_hi} artifact≈{truth_lo}–{truth_hi}",
			)
		)
	else:
		findings.append(Finding("OK", "Abstract", "Teacher macro-F1 range statement matches teacher artifacts (rounded to 3 d.p.)."))

	return findings


def audit_teacher_alignment_example(report_text: str) -> List[Finding]:
	findings: List[Finding] = []
	aln = _load_json(
		REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "RN18_resnet18_seed1337_stageA_img224"
		/ "alignmentreport.json"
	)
	truth_val_rows = int(aln["data"]["val_rows"])

	m = re.search(r"val_rows\s*=\s*([0-9,]+)", report_text)
	if not m:
		findings.append(Finding("WARN", "4.2 Teacher metrics", "Could not find 'val_rows = ...' example in report to verify."))
		return findings

	rep_val_rows = int(m.group(1).replace(",", ""))
	if rep_val_rows != truth_val_rows:
		findings.append(Finding("FAIL", "4.2 Teacher metrics", f"val_rows example mismatch: report={rep_val_rows} artifact={truth_val_rows}"))
	else:
		findings.append(Finding("OK", "4.2 Teacher metrics", "val_rows example matches RN18 alignmentreport.json."))

	return findings


def audit_teacher_per_class(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	_, headers, rows = find_table_by_header(lines, "| Model | Angry | Disgust | Fear | Happy")
	expected_files = {
		"RN18": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "RN18_resnet18_seed1337_stageA_img224"
		/ "reliabilitymetrics.json",
		"B3": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224"
		/ "reliabilitymetrics.json",
		"CNXT": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "CNXT_convnext_tiny_seed1337_stageA_img224"
		/ "reliabilitymetrics.json",
	}

	class_cols = [h for h in headers if h != "Model"]
	for row in rows:
		model = row.get("Model", "").strip()
		if model not in expected_files:
			continue
		data = _load_json(expected_files[model])
		per = data["raw"]["per_class_f1"]
		for cls in class_cols:
			report_str = row.get(cls, "")
			truth = float(per[cls])
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.2 Teacher per-class",
						f"Mismatch {model} {cls}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.2 Teacher per-class") for f in findings):
		findings.append(Finding("OK", "4.2 Teacher per-class", "Teacher per-class F1 table matches reliabilitymetrics.json."))
	return findings


def audit_student_per_class(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	_, headers, rows = find_table_by_header(lines, "| Stage | Angry | Disgust | Fear | Happy")

	paths = {
		"CE": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "CE"
		/ "mobilenetv3_large_100_img224_seed1337_CE_20251223_225031"
		/ "reliabilitymetrics.json",
		"KD": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "_archive"
		/ "2025-12-23"
		/ "KD"
		/ "mobilenetv3_large_100_img224_seed1337_KD_20251223_225031"
		/ "mobilenetv3_large_100_img224_seed1337_KD_20251223_225031"
		/ "reliabilitymetrics.json",
		"DKD": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "_archive"
		/ "2025-12-23"
		/ "DKD"
		/ "mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031"
		/ "mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031"
		/ "reliabilitymetrics.json",
	}

	class_cols = [h for h in headers if h != "Stage"]
	for row in rows:
		stage = row.get("Stage", "").strip()
		if stage not in paths:
			continue
		data = _load_json(paths[stage])
		per = data["raw"]["per_class_f1"]
		for cls in class_cols:
			report_str = row.get(cls, "")
			truth = float(per[cls])
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.4 Student per-class",
						f"Mismatch {stage} {cls}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.4 Student per-class") for f in findings):
		findings.append(Finding("OK", "4.4 Student per-class", "Student per-class F1 table matches reliabilitymetrics.json."))
	return findings


def _find_first_table(lines: List[str]) -> Tuple[int, List[str], List[Dict[str, str]]]:
	for idx, line in enumerate(lines):
		if line.strip().startswith("|"):
			headers, rows, _ = parse_markdown_table(lines, idx)
			return idx, headers, rows
	raise FileNotFoundError("No markdown table found")


def audit_compare_tables(report_lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []

	for i, line in enumerate(report_lines):
		m = re.search(r"^Source:\s*`([^`]+)`\s*$", line.strip())
		if not m:
			continue
		source_rel = m.group(1)
		if "_compare" not in source_rel or not source_rel.endswith(".md"):
			continue
		source_path = REPO_ROOT / source_rel
		if not source_path.exists():
			findings.append(Finding("FAIL", "compare table", f"Compare source not found: {source_rel}"))
			continue

		# report table: first table after Source line
		j = i + 1
		while j < len(report_lines) and not report_lines[j].strip().startswith("|"):
			j += 1
		if j >= len(report_lines):
			findings.append(Finding("FAIL", "compare table", f"No table found after Source: `{source_rel}`"))
			continue

		rep_headers, rep_rows, _ = parse_markdown_table(report_lines, j)

		src_lines = _read_text(source_path).splitlines()
		_, src_headers, src_rows = _find_first_table(src_lines)
		src_header_set = set(src_headers)

		numeric_cols = [
			"Raw acc",
			"Raw macro-F1",
			"Raw ECE",
			"Raw NLL",
			"TS ECE",
			"TS NLL",
			"Minority-F1 (lowest-3)",
		]
		cols_to_check = [c for c in numeric_cols if c in rep_headers and c in src_header_set]
		if not cols_to_check:
			# Nothing numeric overlaps; skip.
			continue

		# Build index of source rows by run-id token (KD_YYYYMMDD_HHMMSS / DKD_...)
		token_re = re.compile(r"(?:KD|DKD)_[0-9]{8}_[0-9]{6}")
		src_index: Dict[str, Dict[str, str]] = {}
		for sr in src_rows:
			hay = " ".join(str(v) for v in sr.values())
			for tok in token_re.findall(hay):
				src_index[tok] = sr

		for rr in rep_rows:
			# Extract run token from the row (prefer the 'Run' column if present)
			row_text = rr.get("Run", "") if "Run" in rr else " ".join(str(v) for v in rr.values())
			toks = token_re.findall(row_text)
			if not toks:
				# If a run token isn't present, we can't verify this row against a compare artifact.
				continue
			tok = toks[0]
			sr = src_index.get(tok)
			if sr is None:
				findings.append(Finding("FAIL", "compare table", f"Could not map report row to compare artifact `{source_rel}` by token {tok}"))
				continue

			for col in cols_to_check:
				report_str = rr.get(col, "")
				truth_str = sr.get(col, "")
				truth = _to_float(str(truth_str))
				if truth is None:
					findings.append(Finding("FAIL", "compare table", f"Compare artifact `{source_rel}` has non-numeric {col} for {tok}: '{truth_str}'"))
					continue
				if not report_matches_artifact(str(report_str), truth):
					findings.append(
						Finding(
							"FAIL",
							"compare table",
							f"Mismatch `{source_rel}` {tok} {col}: report={report_str} compare={truth_str}",
						)
					)

	if not any(f.level == "FAIL" and f.where == "compare table" for f in findings):
		findings.append(Finding("OK", "compare table", "All in-report tables with a `_compare*.md` Source match their compare artifacts."))
	return findings


def audit_teacher_table(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []

	_, _, rows = find_table_by_header(lines, "| Model | Accuracy | Macro-F1")
	expected_files = {
		"RN18": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "RN18_resnet18_seed1337_stageA_img224"
		/ "reliabilitymetrics.json",
		"B3": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224"
		/ "reliabilitymetrics.json",
		"CNXT": REPO_ROOT
		/ "outputs"
		/ "teachers"
		/ "CNXT_convnext_tiny_seed1337_stageA_img224"
		/ "reliabilitymetrics.json",
	}

	for row in rows:
		model = row.get("Model", "").strip()
		if model not in expected_files:
			continue
		data = _load_json(expected_files[model])
		raw = data["raw"]
		ts = data["temperature_scaled"]
		checks = [
			("Accuracy", row["Accuracy"], float(raw["accuracy"])),
			("Macro-F1", row["Macro-F1"], float(raw["macro_f1"])),
			("Raw NLL", row["Raw NLL"], float(raw["nll"])),
			("TS NLL", row["TS NLL"], float(ts["nll"])),
			("Raw ECE", row["Raw ECE"], float(raw["ece"])),
			("TS ECE", row["TS ECE"], float(ts["ece"])),
		]
		for metric, report_str, truth in checks:
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.2 Teacher metrics",
						f"Mismatch {model} {metric}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.2") for f in findings):
		findings.append(Finding("OK", "4.2 Teacher metrics", "Teacher summary table matches teacher reliabilitymetrics.json values."))

	return findings


def audit_ensemble(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []

	ensemble_path = (
		REPO_ROOT
		/ "outputs"
		/ "softlabels"
		/ "_archive"
		/ "bad_list_20251223_121501"
		/ "_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523"
		/ "ensemble_metrics.json"
	)
	data = _load_json(ensemble_path)

	# parse per-class table
	_, headers, rows = find_table_by_header(lines, "| Angry | Disgust | Fear | Happy")
	if rows:
		r0 = rows[0]
		for cls in headers:
			report_str = r0.get(cls, "")
			truth = float(data["per_class_f1"][cls])
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.3 Ensemble per-class",
						f"Mismatch {cls}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.3") for f in findings):
		findings.append(Finding("OK", "4.3 Ensemble", "Ensemble per-class table matches ensemble_metrics.json."))

	return findings


def audit_student_table(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	_, _, rows = find_table_by_header(lines, "| Student stage | Accuracy | Macro-F1")

	paths = {
		"CE": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "CE"
		/ "mobilenetv3_large_100_img224_seed1337_CE_20251223_225031"
		/ "reliabilitymetrics.json",
		"KD": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "_archive"
		/ "2025-12-23"
		/ "KD"
		/ "mobilenetv3_large_100_img224_seed1337_KD_20251223_225031"
		/ "mobilenetv3_large_100_img224_seed1337_KD_20251223_225031"
		/ "reliabilitymetrics.json",
		"DKD": REPO_ROOT
		/ "outputs"
		/ "students"
		/ "_archive"
		/ "2025-12-23"
		/ "DKD"
		/ "mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031"
		/ "mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031"
		/ "reliabilitymetrics.json",
	}

	for row in rows:
		stage = row.get("Student stage", "").strip()
		if stage not in paths:
			continue
		data = _load_json(paths[stage])
		raw = data["raw"]
		ts = data["temperature_scaled"]
		checks = [
			("Accuracy", row["Accuracy"], float(raw["accuracy"])),
			("Macro-F1", row["Macro-F1"], float(raw["macro_f1"])),
			("Raw NLL", row["Raw NLL"], float(raw["nll"])),
			("TS NLL", row["TS NLL"], float(ts["nll"])),
			("Raw ECE", row["Raw ECE"], float(raw["ece"])),
			("TS ECE", row["TS ECE"], float(ts["ece"])),
		]
		for metric, report_str, truth in checks:
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.4 Student metrics",
						f"Mismatch {stage} {metric}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.4") for f in findings):
		findings.append(Finding("OK", "4.4 Student metrics", "Student CE/KD/DKD table matches reliabilitymetrics.json."))

	return findings


def audit_webcam_tables(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	_, _, rows = find_table_by_header(lines, "| Run | Raw acc | Raw macro-F1")

	run_to_json = {
		"20260126_205446": REPO_ROOT / "demo" / "outputs" / "20260126_205446" / "score_results.json",
		"20260126_215903": REPO_ROOT / "demo" / "outputs" / "20260126_215903" / "score_results.json",
	}
	for row in rows:
		run = row.get("Run", "").strip()
		if run not in run_to_json:
			continue
		data = _load_json(run_to_json[run])
		raw = data["metrics"]["raw"]
		sm = data["metrics"]["smoothed"]
		checks = [
			("Raw acc", row["Raw acc"], float(raw["accuracy"])),
			("Raw macro-F1 (present)", row["Raw macro-F1 (present)"], float(raw["macro_f1_present"])),
			("Raw minority-F1 (lowest-3)", row["Raw minority-F1 (lowest-3)"], float(raw["minority_f1_lowest3"])),
			("Smoothed acc", row["Smoothed acc"], float(sm["accuracy"])),
			("Smoothed macro-F1 (present)", row["Smoothed macro-F1 (present)"], float(sm["macro_f1_present"])),
			("Smoothed minority-F1 (lowest-3)", row["Smoothed minority-F1 (lowest-3)"], float(sm["minority_f1_lowest3"])),
		]
		for metric, report_str, truth in checks:
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.6 Webcam scoring",
						f"Mismatch run {run} {metric}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.6") for f in findings):
		findings.append(Finding("OK", "4.6 Webcam scoring", "Webcam scoring table matches score_results.json."))

	return findings


def audit_eval_only_gate(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	_, _, rows = find_table_by_header(lines, "| Model | Raw acc | Raw macro-F1")
	map_paths = {
		"Baseline (CE20251223)": REPO_ROOT
		/ "outputs"
		/ "evals"
		/ "students"
		/ "_baseline_CE20251223_eval_only_test"
		/ "reliabilitymetrics.json",
		"Head-only FT": REPO_ROOT
		/ "outputs"
		/ "evals"
		/ "students"
		/ "FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358"
		/ "reliabilitymetrics.json",
		"BN-only FT": REPO_ROOT
		/ "outputs"
		/ "evals"
		/ "students"
		/ "FT_webcam_bn_20260126_1_eval_only_test"
		/ "reliabilitymetrics.json",
	}

	for row in rows:
		label = row.get("Model", "").strip()
		if label not in map_paths:
			continue
		data = _load_json(map_paths[label])
		raw = data["raw"]
		ts = data["temperature_scaled"]
		checks = [
			("Raw acc", row["Raw acc"], float(raw["accuracy"])),
			("Raw macro-F1", row["Raw macro-F1"], float(raw["macro_f1"])),
			("TS ECE", row["TS ECE"], float(ts["ece"])),
			("TS NLL", row["TS NLL"], float(ts["nll"])),
		]
		for metric, report_str, truth in checks:
			if not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.7 Offline gate",
						f"Mismatch {label} {metric}: report={report_str} artifact={truth}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.7") for f in findings):
		findings.append(Finding("OK", "4.7 Offline gate", "Offline safety gate table matches eval-only reliabilitymetrics.json."))

	return findings


def audit_expw_compare(lines: List[str]) -> List[Finding]:
	findings: List[Finding] = []
	compare_path = REPO_ROOT / "outputs" / "evals" / "_compare_20260119_170620_domainshift_expw_full_manifest_test.md"
	compare_text = _read_text(compare_path)
	comp_lines = compare_text.splitlines()
	_, _, comp_rows = find_table_by_header(comp_lines, "| Label | Mode | Epochs")

	_, _, rep_rows = find_table_by_header(lines, "| Label | Mode | Epochs")

	comp_by_label = {r["Label"].strip(): r for r in comp_rows}
	for r in rep_rows:
		label = r["Label"].strip()
		if label not in comp_by_label:
			findings.append(Finding("FAIL", "4.8 ExpW compare", f"Label not found in compare artifact: {label}"))
			continue
		src = comp_by_label[label]
		for col in [
			"Raw acc",
			"Raw macro-F1",
			"Raw ECE",
			"Raw NLL",
			"TS ECE",
			"TS NLL",
			"Minority-F1 (lowest-3)",
		]:
			report_str = r.get(col, "")
			truth_str = src.get(col, "")
			truth = _to_float(truth_str)
			if truth is None or not report_matches_artifact(report_str, truth):
				findings.append(
					Finding(
						"FAIL",
						"4.8 ExpW compare",
						f"Mismatch {label} {col}: report={report_str} compare={truth_str}",
					)
				)

	if not any(f.level == "FAIL" and f.where.startswith("4.8") for f in findings):
		findings.append(Finding("OK", "4.8 ExpW compare", "ExpW cross-dataset table matches the compare artifact."))

	return findings


def main() -> None:
	if not REPORT_PATH.exists():
		raise SystemExit(f"Report not found: {REPORT_PATH}")

	report_text = _read_text(REPORT_PATH)
	lines = report_text.splitlines()

	findings: List[Finding] = []
	findings.extend(audit_paths(report_text))
	findings.extend(audit_dataset_counts(report_text))
	findings.extend(audit_abstract_teacher_range(report_text))
	findings.extend(audit_teacher_alignment_example(report_text))
	findings.extend(audit_teacher_table(lines))
	findings.extend(audit_teacher_per_class(lines))
	findings.extend(audit_ensemble(lines))
	findings.extend(audit_student_table(lines))
	findings.extend(audit_student_per_class(lines))
	findings.extend(audit_webcam_tables(lines))
	findings.extend(audit_eval_only_gate(lines))
	findings.extend(audit_expw_compare(lines))
	findings.extend(audit_compare_tables(lines))

	fails = [f for f in findings if f.level == "FAIL"]
	warns = [f for f in findings if f.level == "WARN"]
	oks = [f for f in findings if f.level == "OK"]

	print("=== Final report audit ===")
	print(f"OK: {len(oks)} | WARN: {len(warns)} | FAIL: {len(fails)}")
	print()

	for f in fails:
		print(f"FAIL [{f.where}] {f.message}")

	for f in warns:
		print(f"WARN [{f.where}] {f.message}")

	if not fails:
		print("\nNo numeric mismatches found for audited tables.")


if __name__ == "__main__":
	main()