param(
  [string]$ZipPath = "outputs\\realtime_fer_backup.zip",
  [string]$StageDir = "outputs\\_realtime_fer_backup_stage"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoName = Split-Path -Leaf $RepoRoot

# Prefer repo venv if present; otherwise fall back to PATH python.
$PythonExe = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (!(Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }

$PickScript = Join-Path $RepoRoot "scripts\\pick_best_student_ckpt.py"
if (!(Test-Path -LiteralPath $PickScript)) { throw "Missing: $PickScript" }

$OnnxExportScript = Join-Path $RepoRoot "scripts\\export_student_onnx.py"
if (!(Test-Path -LiteralPath $OnnxExportScript)) { throw "Missing: $OnnxExportScript" }

Write-Host "Repo:   $RepoRoot"
Write-Host "Python: $PythonExe"

$bestJson = & $PythonExe $PickScript
$best = $bestJson | ConvertFrom-Json
if ($best.error) { throw "Failed to pick best student checkpoint: $($best.error)" }

$bestCkptRel = [string]$best.ckpt
$bestRunDirRel = Split-Path -Parent $bestCkptRel

$bestCkptAbs = Join-Path $RepoRoot $bestCkptRel
$bestRunDirAbs = Join-Path $RepoRoot $bestRunDirRel

if (!(Test-Path -LiteralPath $bestCkptAbs)) { throw "Checkpoint not found: $bestCkptAbs" }

$StageAbs = Join-Path $RepoRoot $StageDir
$PkgRoot = Join-Path $StageAbs $RepoName

if (Test-Path -LiteralPath $StageAbs) { Remove-Item -LiteralPath $StageAbs -Recurse -Force }
New-Item -ItemType Directory -Force -Path $PkgRoot | Out-Null

function Copy-Dir([string]$rel) {
  $src = Join-Path $RepoRoot $rel
  $dst = Join-Path $PkgRoot $rel
  if (!(Test-Path -LiteralPath $src)) { throw "Missing required folder: $src" }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
}

function Copy-File([string]$rel) {
  $src = Join-Path $RepoRoot $rel
  $dst = Join-Path $PkgRoot $rel
  if (!(Test-Path -LiteralPath $src)) { throw "Missing required file: $src" }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  Copy-Item -LiteralPath $src -Destination $dst -Force
}

# 1) Code needed to run real-time FER
Copy-Dir "demo"
Copy-Dir "src"
Copy-Dir "tools"

# 2) Minimal scripts used by demo/runtime and reproducibility tools
Copy-File "scripts\\train_teacher.py"          # transforms
Copy-File "scripts\\train_student.py"          # metrics helpers used by eval script
Copy-File "scripts\\realtime_infer_arcface.py" # wrapper
Copy-File "scripts\\eval_student_checkpoint.py" # domain shift eval
Copy-File "scripts\\pick_best_student_ckpt.py" # model selection
Copy-File "scripts\\export_student_onnx.py"     # ONNX export
Copy-File "scripts\\score_live_results.py"       # summarize demo logs (if used)
Copy-File "scripts\\check_device.py"

# 3) Docs + requirements for reproducibility
if (Test-Path -LiteralPath (Join-Path $RepoRoot "docs\\realtime_demo_playbook.md")) {
  Copy-File "docs\\realtime_demo_playbook.md"
}
if (Test-Path -LiteralPath (Join-Path $RepoRoot "docs\\realtime_demo_usb_pack.md")) {
  Copy-File "docs\\realtime_demo_usb_pack.md"
}
if (Test-Path -LiteralPath (Join-Path $RepoRoot "requirements.txt")) {
  Copy-File "requirements.txt"
}
if (Test-Path -LiteralPath (Join-Path $RepoRoot "requirements-directml.txt")) {
  Copy-File "requirements-directml.txt"
}

# Optional lightweight provenance artifacts
if (Test-Path -LiteralPath (Join-Path $RepoRoot "table.md")) {
  Copy-File "table.md"
}
if (Test-Path -LiteralPath (Join-Path $RepoRoot "outputs\\training_data_counts.json")) {
  Copy-File "outputs\\training_data_counts.json"
}

# Remove demo output artifacts (keep folder empty so demo can write)
$demoOutputs = Join-Path $PkgRoot "demo\\outputs"
if (Test-Path -LiteralPath $demoOutputs) {
  Remove-Item -LiteralPath $demoOutputs -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $demoOutputs | Out-Null

# 4) Models: copy the selected student run (best.pt + JSONs) and export ONNX
$dstRunDirAbs = Join-Path $PkgRoot $bestRunDirRel
New-Item -ItemType Directory -Force -Path $dstRunDirAbs | Out-Null

Copy-Item -LiteralPath $bestCkptAbs -Destination (Join-Path $dstRunDirAbs "best.pt") -Force

foreach ($name in @("calibration.json", "reliabilitymetrics.json", "history.json")) {
  $src = Join-Path $bestRunDirAbs $name
  if (Test-Path -LiteralPath $src) {
    Copy-Item -LiteralPath $src -Destination (Join-Path $dstRunDirAbs $name) -Force
  }
}

# Export ONNX to a convenient path
$modelsDir = Join-Path $PkgRoot "models"
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

$onnxOut = Join-Path $modelsDir "student_best.onnx"
$onnxMeta = Join-Path $modelsDir "student_best_onnx_export_meta.json"

Write-Host "Exporting student ONNX..."
$exportMetaJson = & $PythonExe $OnnxExportScript --checkpoint $bestCkptAbs --out $onnxOut --meta-out $onnxMeta
Write-Host $exportMetaJson

# Create a simple README inside the package
$readme = @()
$readme += "# Real-time FER backup pack"
$readme += ""
$readme += "This ZIP contains the minimum code + tools + model artifacts needed to run the real-time FER demo and reproduce key evaluation outputs." 
$readme += ""
$readme += "## Included"
$readme += "- demo/: real-time UI + face detector logic"
$readme += "- src/: FER utilities"
$readme += "- tools/: diagnostics/helpers"
$readme += "- scripts/: runtime wrapper + eval + ONNX exporter"
$readme += "- Selected student run: $bestRunDirRel\\best.pt (+ calibration/metrics when present)"
$readme += "- ONNX export: models\\student_best.onnx"
$readme += ""
$readme += "## Quick start (after unzip)"
$readme += "1) Create venv and install deps:"
$readme += '```powershell'
$readme += "python -m venv .venv"
$readme += ".\\.venv\\Scripts\\python.exe -m pip install --upgrade pip"
$readme += ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
$readme += '```'
$readme += "2) Run the demo (student):"
$readme += '```powershell'
$readme += "python demo\\realtime_demo.py --model-kind student --model-ckpt $bestCkptRel"
$readme += '```'
$readme += ""
$readme += "## Provenance"
$readme += "- This pack is created from the repo at packaging time; install dependencies from requirements files." 
$readme += ""

$readmePath = Join-Path $PkgRoot "BACKUP_README.md"
$readme | Out-File -FilePath $readmePath -Encoding utf8

# Capture pip freeze (best-effort)
try {
  $freezePath = Join-Path $PkgRoot "pip_freeze.txt"
  & $PythonExe -m pip freeze | Out-File -FilePath $freezePath -Encoding utf8
} catch {
  Write-Host "WARN: pip freeze failed: $($_.Exception.Message)"
}

# Ensure outputs/students exists
New-Item -ItemType Directory -Force -Path (Join-Path $PkgRoot "outputs\\students") | Out-Null

# Zip
$ZipAbs = Join-Path $RepoRoot $ZipPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ZipAbs) | Out-Null
if (Test-Path -LiteralPath $ZipAbs) { Remove-Item -LiteralPath $ZipAbs -Force }

Compress-Archive -Path (Join-Path $StageAbs "*") -DestinationPath $ZipAbs -Force

Write-Host "\nCreated ZIP: $ZipAbs"
Write-Host "Included best student ckpt: $bestCkptRel"
Write-Host ("Score: macro_f1={0:N6} accuracy={1:N6}" -f $best.raw.macro_f1, $best.raw.accuracy)
