param(
    [ValidateSet('A', 'B', 'AthenB')]
    [string]$Stage = 'AthenB',

    [int]$StageAImageSize = 224,
    [int]$StageBImageSize = 384,

    [int]$BatchSize = 64,
    [int]$Epochs = 60,
    [int]$NumWorkers = 8,
    [int]$Seed = 1337,
    [int]$AccumSteps = 1,
    [int]$EvalEvery = 1,

    # Set this on multi-GPU servers to pin this run to one GPU.
    # Example: -CudaDevice 0
    [string]$CudaDevice = '',

    [switch]$Clean,
    [switch]$CleanStageA,
    [switch]$CleanStageB,

    [switch]$UnlockStale,
    [switch]$ForceUnlock,

    [switch]$NoClahe,
    [switch]$SkipOnnxDuringTrain,
    [switch]$Smoke,

    # Which manifest to use for training/eval splits.
    # - classification_manifest_hq_train.csv: curated HQ training mix
    # - classification_manifest.csv: full unified manifest (recommended when reproducing old RAF-DB test numbers)
    [ValidateSet('hq', 'full')]
    [string]$ManifestPreset = 'full'

    ,
    # Optional suffix appended to output dir names so you can run multiple variants
    # without overwriting previous runs (e.g., 'pretrained_true_v1').
    [string]$OutSuffix = ''

    ,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

if ($Help) {
    Write-Host "B3 two-stage trainer (Stage A 224 include FERPlus; Stage B 384 exclude FERPlus)."
    Write-Host "Examples:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_teachers_overnight_b3_2stage.ps1 -Stage AthenB -ManifestPreset full -CudaDevice 0 -EvalEvery 1"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_teachers_overnight_b3_2stage.ps1 -Stage B -ManifestPreset full -CudaDevice 0 -UnlockStale"
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
$LogTag = if ($OutSuffix -and $OutSuffix.Trim().Length -gt 0) { "B3_${OutSuffix}_$Stamp" } else { "B3_" + $Stamp }
$LogDir = Join-Path $RepoRoot (Join-Path 'outputs\teachers' (Join-Path '_overnight_logs_2stage' $LogTag))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Two-stage policy:
# - Stage A (224): include FERPlus
# - Stage B (384): exclude FERPlus, init from Stage A checkpoint
$StageAInclude = 'ferplus,rafdb_basic,affectnet_full_balanced,expw_hq'
$StageBExclude = 'ferplus'

$tag = 'B3'
$name = 'tf_efficientnet_b3'

Write-Host "Logs: $LogDir"
Write-Host "Stage=$Stage Model=$name ManifestPreset=$ManifestPreset BatchSize=$BatchSize Epochs=$Epochs NumWorkers=$NumWorkers EvalEvery=$EvalEvery Smoke=$Smoke"

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

function Invoke-TeacherTrain {
    param(
        [string]$StageName,
        [int]$ImageSize,
        [string]$OutputDir,
        [string]$LogPath,
        [string]$IncludeSources,
        [string]$ExcludeSources,
        [string]$InitFrom
    )

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
        '--output-dir', $OutputDir
    )

    if ($IncludeSources -and $IncludeSources.Trim().Length -gt 0) {
        $cliArgs += @('--include-sources', $IncludeSources)
    }

    if ($ExcludeSources -and $ExcludeSources.Trim().Length -gt 0) {
        $cliArgs += @('--exclude-sources', $ExcludeSources)
    }

    if ($InitFrom -and $InitFrom.Trim().Length -gt 0) {
        $cliArgs += @('--init-from', $InitFrom)
    }

    if (-not $NoClahe) {
        $cliArgs += @('--clahe')
    }

    if ($SkipOnnxDuringTrain) {
        $cliArgs += @('--skip-onnx-during-train')
    }

    if ($Smoke) {
        $cliArgs += @('--smoke')
    }

    Write-Host "\n=== [$tag] $name | $StageName | img=$ImageSize ==="
    Write-Host "OutputDir: $OutputDir"
    Write-Host "Logging to: $LogPath"

    if (Test-RunLocked -Dir $OutputDir) {
        if ($UnlockStale) {
            Clear-RunLockIfStale -Dir $OutputDir -Force:$ForceUnlock
        }
        if (Test-RunLocked -Dir $OutputDir) {
            $lockPath = Join-Path $OutputDir '.run.lock'
            throw "Output dir is locked by a running process: $lockPath"
        }
    }

    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $PythonExe @cliArgs 2>&1 | Tee-Object -FilePath $LogPath
    $ErrorActionPreference = $prevEap

    if ($LASTEXITCODE -ne 0) {
        throw "Teacher run failed for $name ($StageName) (exit code $LASTEXITCODE). See log: $LogPath"
    }

    $required = @('alignmentreport.json', 'history.json', 'reliabilitymetrics.json', 'calibration.json', 'checkpoint_last.pt')
    foreach ($f in $required) {
        $p = Join-Path $OutputDir $f
        if (!(Test-Path $p)) {
            throw "Missing artifact: $p"
        }
    }
}

$baseName = $tag + "_" + $name + "_seed" + $Seed
if ($OutSuffix -and $OutSuffix.Trim().Length -gt 0) {
    $baseName = $baseName + "_" + $OutSuffix.Trim()
}

$baseOut = Join-Path $RepoRoot ("outputs\teachers\" + $baseName)

if ($Stage -eq 'A' -or $Stage -eq 'AthenB') {
    $outA = $baseOut + "_stageA_img" + $StageAImageSize
    $logA = Join-Path $LogDir ("$tag`_$name`_stageA_img$StageAImageSize.log")

    if ($Clean -or $CleanStageA) {
        Write-Host "Cleaning Stage A output: $outA"
        Remove-DirSafe -Dir $outA
    }

    Invoke-TeacherTrain -StageName 'StageA' -ImageSize $StageAImageSize -OutputDir $outA -LogPath $logA -IncludeSources $StageAInclude -ExcludeSources '' -InitFrom ''
}

if ($Stage -eq 'B' -or $Stage -eq 'AthenB') {
    $outB = $baseOut + "_stageB_img" + $StageBImageSize
    $logB = Join-Path $LogDir ("$tag`_$name`_stageB_img$StageBImageSize.log")

    if ($Clean -or $CleanStageB) {
        Write-Host "Cleaning Stage B output: $outB"
        Remove-DirSafe -Dir $outB
    }

    $init = ''
    $outA = $baseOut + "_stageA_img" + $StageAImageSize
    $ckptA_best = Join-Path $outA 'best.pt'
    $ckptA_last = Join-Path $outA 'checkpoint_last.pt'

    $ckptB = Join-Path $outB 'checkpoint_last.pt'
    if (!(Test-Path $ckptB)) {
        if (Test-Path $ckptA_best) {
            $init = $ckptA_best
        } elseif (Test-Path $ckptA_last) {
            $init = $ckptA_last
        } else {
            throw "Stage B requested but no Stage B checkpoint found ($ckptB) and no Stage A checkpoint found ($ckptA_best or $ckptA_last)."
        }
    }

    Invoke-TeacherTrain -StageName 'StageB' -ImageSize $StageBImageSize -OutputDir $outB -LogPath $logB -IncludeSources '' -ExcludeSources $StageBExclude -InitFrom $init
}

Write-Host "\nB3 two-stage run completed."
