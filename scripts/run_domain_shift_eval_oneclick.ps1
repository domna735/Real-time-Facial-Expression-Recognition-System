param(
  [string]$EvalManifest = "Training_data_cleaned/expw_full_manifest.csv",
  [ValidateSet("val","test")][string]$EvalSplit = "test",
  [int]$BatchSize = 256,
  [int]$NumWorkers = 4,
  [int]$MaxBatches = 0,
  [switch]$UseAmp,

  # Optional explicit list of run dirs to evaluate. If omitted, uses the latest DKD candidates from the Jan 1 sweep.
  [string[]]$RunDirs = @(
    "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722",
    "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953",
    "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203",
    "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949",
    "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602"
  )
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$p) {
  $full = Join-Path -Path (Get-Location) -ChildPath $p
  return (Resolve-Path -LiteralPath $full).Path
}

function Pick-Checkpoint([string]$runDir) {
  $best = Join-Path $runDir "best.pt"
  if (Test-Path -LiteralPath $best) { return $best }
  $last = Join-Path $runDir "checkpoint_last.pt"
  if (Test-Path -LiteralPath $last) { return $last }
  throw "No checkpoint found in $runDir (expected best.pt or checkpoint_last.pt)"
}

function Get-LeafBase([string]$p) {
  return [System.IO.Path]::GetFileNameWithoutExtension($p)
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalManifestPath = Resolve-RepoPath $EvalManifest

$outRoot = Join-Path (Resolve-RepoPath "outputs") "evals\students"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$evalDirs = @()

foreach ($rd in $RunDirs) {
  $runDirAbs = Resolve-RepoPath $rd
  $ckpt = Pick-Checkpoint $runDirAbs

  $runName = Split-Path -Leaf $runDirAbs
  $outDir = Join-Path $outRoot ("{0}__{1}__{2}__{3}" -f $runName, (Get-LeafBase $evalManifestPath), $EvalSplit, $stamp)

  Write-Host "[eval] $runName -> $EvalSplit on $(Split-Path -Leaf $evalManifestPath)" -ForegroundColor Cyan

  $py = Resolve-RepoPath ".venv/Scripts/python.exe"
  $cmd = @(
    $py,
    (Resolve-RepoPath "scripts/eval_student_checkpoint.py"),
    "--checkpoint", $ckpt,
    "--eval-manifest", $evalManifestPath,
    "--eval-split", $EvalSplit,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--max-batches", $MaxBatches,
    "--out-dir", $outDir
  )
  if ($UseAmp) { $cmd += "--use-amp" }

  & $cmd[0] $cmd[1..($cmd.Length-1)]

  $evalDirs += $outDir
}

$compareOut = Join-Path (Resolve-RepoPath "outputs") ("evals/_compare_{0}_domainshift_{1}_{2}.md" -f $stamp, (Get-LeafBase $evalManifestPath), $EvalSplit)

Write-Host "[compare] Writing: $compareOut" -ForegroundColor Green

$py2 = Resolve-RepoPath ".venv/Scripts/python.exe"
& $py2 (Resolve-RepoPath "tools/diagnostics/compare_student_runs.py") @($evalDirs) "--out" $compareOut

Write-Host "Done." -ForegroundColor Green
