param(
    [int]$ImageSize = 224,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 4,
    [int]$Seed = 1337,

    [int]$EpochsHead = 3,
    [int]$EpochsLastBlock = 5,

    [double]$LrHead = 1e-3,
    [double]$LrLastBlock = 3e-4,

    [switch]$NoClahe,
    [switch]$UseAmp,
    [switch]$SkipPostEval
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$TrainScript = Join-Path $RepoRoot 'scripts\train_student.py'
$Manifest = Join-Path $RepoRoot 'Training_data_cleaned\classification_manifest_hq_train.csv'
$DataRoot = Join-Path $RepoRoot 'Training_data_cleaned'

if (!(Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
if (!(Test-Path $TrainScript)) { throw "Script not found: $TrainScript" }
if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
if (!(Test-Path $DataRoot)) { throw "Data root not found: $DataRoot" }

function Get-InitCheckpointPath([string]$OutDir) {
    $best = Join-Path $OutDir 'best.pt'
    if (Test-Path $best) { return $best }

    $last = Join-Path $OutDir 'checkpoint_last.pt'
    if (Test-Path $last) { return $last }

    throw "No checkpoint found in: $OutDir"
}

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$RunRoot = Join-Path $RepoRoot (Join-Path 'outputs\students' (Join-Path '_paper_resnet50_transfer' $Stamp))
$LogDir = Join-Path $RunRoot '_logs'

New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ModelName = 'resnet50'
$Stage1Out = Join-Path $RunRoot 'stage1_head'
$Stage2Out = Join-Path $RunRoot 'stage2_lastblock_head'

$stage1Log = Join-Path $LogDir 'stage1_head.log'
$stage2Log = Join-Path $LogDir 'stage2_lastblock_head.log'

Write-Host "RunRoot: $RunRoot"
Write-Host "Model=$ModelName ImageSize=$ImageSize BatchSize=$BatchSize Seed=$Seed"
Write-Host "Stage1: tune=head epochs=$EpochsHead lr=$LrHead"
Write-Host "Stage2: tune=lastblock_head epochs=$EpochsLastBlock lr=$LrLastBlock"

# --- Stage 1: head-only ---
$args1 = @(
    $TrainScript,
    '--mode', 'ce',
    '--model', $ModelName,
    '--manifest', $Manifest,
    '--data-root', $DataRoot,
    '--image-size', $ImageSize,
    '--batch-size', $BatchSize,
    '--num-workers', $NumWorkers,
    '--seed', $Seed,
    '--epochs', $EpochsHead,
    '--lr', $LrHead,
    '--tune', 'head',
    '--output-dir', $Stage1Out
)

if (-not $NoClahe) {
    $args1 += @('--use-clahe')
}

if ($UseAmp) {
    $args1 += @('--use-amp')
}

Write-Host "\n=== Stage 1: head-only fine-tune ==="
Write-Host "Logging to: $stage1Log"
& $PythonExe @args1 2>&1 | Tee-Object -FilePath $stage1Log

# --- Stage 2: last-block + head ---
$initFrom = Get-InitCheckpointPath -OutDir $Stage1Out
Write-Host "\nInit-from checkpoint: $initFrom"

$args2 = @(
    $TrainScript,
    '--mode', 'ce',
    '--model', $ModelName,
    '--manifest', $Manifest,
    '--data-root', $DataRoot,
    '--image-size', $ImageSize,
    '--batch-size', $BatchSize,
    '--num-workers', $NumWorkers,
    '--seed', $Seed,
    '--epochs', $EpochsLastBlock,
    '--lr', $LrLastBlock,
    '--tune', 'lastblock_head',
    '--init-from', $initFrom,
    '--output-dir', $Stage2Out
)

if (-not $NoClahe) {
    $args2 += @('--use-clahe')
}

if ($UseAmp) {
    $args2 += @('--use-amp')
}

if (-not $SkipPostEval) {
    $args2 += @('--post-eval')
}

Write-Host "\n=== Stage 2: lastblock+head fine-tune ==="
Write-Host "Logging to: $stage2Log"
& $PythonExe @args2 2>&1 | Tee-Object -FilePath $stage2Log

Write-Host "\nDone. Outputs:"
Write-Host "  Stage1: $Stage1Out"
Write-Host "  Stage2: $Stage2Out"
Write-Host "  Logs:   $LogDir"
Write-Host "\nIf --post-eval was enabled, eval artifacts are under: outputs\evals\students\";
Write-Host "(search by this run's stamp or by checkpoint filename)."