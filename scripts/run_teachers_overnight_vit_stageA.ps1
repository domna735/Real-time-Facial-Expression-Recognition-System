param(
    [int]$ImageSize = 224,

    [int]$BatchSize = 64,
    [int]$Epochs = 60,
    [int]$NumWorkers = 8,
    [int]$Seed = 1337,
    [int]$AccumSteps = 1,
    [int]$EvalEvery = 1,

    [string]$CudaDevice = '',

    [switch]$Clean,

    [switch]$UnlockStale,
    [switch]$ForceUnlock,

    [switch]$NoClahe,
    [switch]$SkipOnnxDuringTrain,
    [switch]$Smoke,

    [ValidateSet('hq', 'full')]
    [string]$ManifestPreset = 'full',

    # Optional suffix appended to output dir names so you can run multiple variants
    # without overwriting previous runs.
    [string]$OutSuffix = '',

    [switch]$Help
)

$ErrorActionPreference = 'Stop'

if ($Help) {
    Write-Host "ViT Stage-A-only trainer (img=$ImageSize, includes FERPlus)."
    Write-Host "Example:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_teachers_overnight_vit_stageA.ps1 -ManifestPreset full -CudaDevice 0 -EvalEvery 1"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_teachers_overnight_vit_stageA.ps1 -ManifestPreset full -CudaDevice 0 -OutSuffix my_variant_1"
    return
}

if ($CudaDevice -and $CudaDevice.Trim().Length -gt 0) {
    $env:CUDA_VISIBLE_DEVICES = $CudaDevice
    Write-Host "CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES)"
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$TrainScript = Join-Path $RepoRoot 'scripts\train_teacher.py'
$Manifest = if ($ManifestPreset -eq 'full') {
    Join-Path $RepoRoot 'Training_data_cleaned\classification_manifest.csv'
} else {
    Join-Path $RepoRoot 'Training_data_cleaned\classification_manifest_hq_train.csv'
}
$OutRoot = Join-Path $RepoRoot 'Training_data_cleaned'

if (!(Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
if (!(Test-Path $TrainScript)) { throw "Script not found: $TrainScript" }
if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogTag = if ($OutSuffix -and $OutSuffix.Trim().Length -gt 0) { "ViT_${OutSuffix}_$Stamp" } else { "ViT_" + $Stamp }
$LogDir = Join-Path $RepoRoot (Join-Path 'outputs\teachers' (Join-Path '_overnight_logs_2stage' $LogTag))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Stage A policy: include FERPlus
$IncludeSources = 'ferplus,rafdb_basic,affectnet_full_balanced,expw_hq'

$tag = 'ViT'
# timm model name
$name = 'vit_base_patch16_384'

Write-Host "Logs: $LogDir"
Write-Host "Stage=A Model=$name ManifestPreset=$ManifestPreset BatchSize=$BatchSize Epochs=$Epochs NumWorkers=$NumWorkers EvalEvery=$EvalEvery Smoke=$Smoke"

if ($BatchSize -lt 14) {
    throw "BatchSize must be >= 14 (7 classes x min_per_class=2 for BalancedBatchSampler). Got: $BatchSize"
}

function Test-RunLocked {
    param([string]$Dir)
    $lockPath = Join-Path $Dir '.run.lock'
    return (Test-Path $lockPath)
}

function Clear-RunLockIfStale {
    param(
        [string]$Dir,
        [switch]$Force
    )

    $lockPath = Join-Path $Dir '.run.lock'
    if (!(Test-Path $lockPath)) { return }

    if ($Force) {
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
        return
    }

    $lockPid = $null
    try {
        $txt = Get-Content -LiteralPath $lockPath -Raw -ErrorAction Stop
        $j = $txt | ConvertFrom-Json -ErrorAction Stop
        $lockPid = $j.pid
    } catch {
        return
    }

    if (-not $lockPid) { return }

    $proc = $null
    try {
        $proc = Get-Process -Id ([int]$lockPid) -ErrorAction Stop
    } catch {
        $proc = $null
    }

    if (-not $proc) {
        Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
    }
}

function Remove-DirSafe {
    param([string]$Dir)

    if (!(Test-Path $Dir)) { return }

    if (Test-RunLocked -Dir $Dir) {
        if ($UnlockStale) {
            Clear-RunLockIfStale -Dir $Dir -Force:$ForceUnlock
        }
        if (Test-RunLocked -Dir $Dir) {
            $lockPath = Join-Path $Dir '.run.lock'
            throw "Refusing to delete output dir because lock exists: $lockPath"
        }
    }

    Remove-Item -LiteralPath $Dir -Recurse -Force
}

$baseName = $tag + "_" + $name + "_seed" + $Seed
if ($OutSuffix -and $OutSuffix.Trim().Length -gt 0) {
    $baseName = $baseName + "_" + $OutSuffix.Trim()
}

$outA = Join-Path $RepoRoot ("outputs\\teachers\\" + $baseName + "_stageA_img" + $ImageSize)
$logA = Join-Path $LogDir ("$tag`_$name`_stageA_img$ImageSize.log")

if ($Clean) {
    Write-Host "Cleaning Stage A output: $outA"
    Remove-DirSafe -Dir $outA
}

$cliArgs = @(
    $TrainScript,
    '--model', $name,
    '--manifest', $Manifest,
    '--out-root', $OutRoot,
    '--image-size', $ImageSize,
    '--batch-size', $BatchSize,
    '--num-workers', $NumWorkers,
    '--seed', $Seed,
    '--accum-steps', $AccumSteps,
    '--eval-every', $EvalEvery,
    '--max-epochs', $Epochs,
    '--min-lr', 1e-5,
    '--checkpoint-every', 10,
    '--output-dir', $outA,
    '--include-sources', $IncludeSources
)

if (-not $NoClahe) {
    $cliArgs += @('--clahe')
}

if ($SkipOnnxDuringTrain) {
    $cliArgs += @('--skip-onnx-during-train')
}

if ($Smoke) {
    $cliArgs += @('--smoke')
}

Write-Host "\n=== [$tag] $name | StageA | img=$ImageSize ==="
Write-Host "OutputDir: $outA"
Write-Host "Logging to: $logA"

if (Test-RunLocked -Dir $outA) {
    if ($UnlockStale) {
        Clear-RunLockIfStale -Dir $outA -Force:$ForceUnlock
    }
    if (Test-RunLocked -Dir $outA) {
        $lockPath = Join-Path $outA '.run.lock'
        throw "Output dir is locked by a running process: $lockPath"
    }
}

$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $PythonExe @cliArgs 2>&1 | Tee-Object -FilePath $logA
$ErrorActionPreference = $prevEap

if ($LASTEXITCODE -ne 0) {
    throw "Teacher run failed for $name (StageA) (exit code $LASTEXITCODE). See log: $logA"
}

$required = @('alignmentreport.json', 'history.json', 'reliabilitymetrics.json', 'calibration.json', 'checkpoint_last.pt')
foreach ($f in $required) {
    $p = Join-Path $outA $f
    if (!(Test-Path $p)) {
        throw "Missing artifact: $p"
    }
}

Write-Host "Done. OutputDir: $outA"
