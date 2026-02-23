param(
  [string]$ZipPath = "outputs\\realtime_demo_usb.zip",
  [string]$StageDir = "outputs\\_usb_realtime_demo_stage"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RepoName = Split-Path -Leaf $RepoRoot

# Prefer repo venv if present; otherwise fall back to PATH python.
$PythonExe = Join-Path $RepoRoot ".venv\\Scripts\\python.exe"
if (!(Test-Path -LiteralPath $PythonExe)) { $PythonExe = "python" }

$PickScript = Join-Path $RepoRoot "scripts\\pick_best_student_ckpt.py"
if (!(Test-Path -LiteralPath $PickScript)) { throw "Missing: $PickScript" }

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

# Core demo runtime
Copy-Dir "demo"
Copy-Dir "src"
Copy-Dir "tools"

# Minimal scripts needed by demo (realtime_demo imports scripts.train_teacher for transforms)
Copy-File "scripts\\train_teacher.py"
Copy-File "scripts\\realtime_infer_arcface.py"
Copy-File "scripts\\check_device.py"

# Requirements (torch/timm install still depends on target machine)
if (Test-Path -LiteralPath (Join-Path $RepoRoot "requirements.txt")) {
  Copy-File "requirements.txt"
}
if (Test-Path -LiteralPath (Join-Path $RepoRoot "requirements-directml.txt")) {
  Copy-File "requirements-directml.txt"
}

# Remove demo output artifacts (keep folder empty so demo can write)
$demoOutputs = Join-Path $PkgRoot "demo\\outputs"
if (Test-Path -LiteralPath $demoOutputs) {
  Remove-Item -LiteralPath $demoOutputs -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $demoOutputs | Out-Null

# Copy only the selected student run artifacts (best.pt + calibration + reliabilitymetrics)
$dstRunDirAbs = Join-Path $PkgRoot $bestRunDirRel
New-Item -ItemType Directory -Force -Path $dstRunDirAbs | Out-Null

Copy-Item -LiteralPath $bestCkptAbs -Destination (Join-Path $dstRunDirAbs "best.pt") -Force

$calAbs = Join-Path $bestRunDirAbs "calibration.json"
if (Test-Path -LiteralPath $calAbs) {
  Copy-Item -LiteralPath $calAbs -Destination (Join-Path $dstRunDirAbs "calibration.json") -Force
}

$metricsAbs = Join-Path $bestRunDirAbs "reliabilitymetrics.json"
if (Test-Path -LiteralPath $metricsAbs) {
  Copy-Item -LiteralPath $metricsAbs -Destination (Join-Path $dstRunDirAbs "reliabilitymetrics.json") -Force
}

# Ensure outputs/students path exists even if someone expects it
New-Item -ItemType Directory -Force -Path (Join-Path $PkgRoot "outputs\\students") | Out-Null

# Zip
$ZipAbs = Join-Path $RepoRoot $ZipPath
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ZipAbs) | Out-Null
if (Test-Path -LiteralPath $ZipAbs) { Remove-Item -LiteralPath $ZipAbs -Force }

Compress-Archive -Path (Join-Path $StageAbs "*") -DestinationPath $ZipAbs -Force

Write-Host "\nCreated ZIP: $ZipAbs"
Write-Host "Included best student ckpt: $bestCkptRel"
Write-Host ("Score: macro_f1={0:N6} accuracy={1:N6}" -f $best.raw.macro_f1, $best.raw.accuracy)
Write-Host "\nUSB run command (after unzip + cd):"
Write-Host ("python demo/realtime_demo.py --model-kind student --model-ckpt {0}" -f $bestCkptRel)
