param(
    # Stage selection:
    #   A = 224 pretrain (includes ferplus)
    #   B = 384 finetune (excludes ferplus, resumes from stage A)
    #   AB = run both sequentially
    [ValidateSet('A','B','AB')][string]$Stage = 'AB',

    [int]$Seed = 1337,

    # Defaults tuned for RTX 5070 Ti + 32GB RAM (adjust if you hit I/O bottlenecks)
    [int]$NumWorkers = 8,
    [int]$BatchSize = 64,
    [int]$AccumSteps = 1,

    [int]$Epochs = 60,
    [int]$EvalEvery = 10,

    [switch]$Clahe,
    [switch]$SkipOnnxDuringTrain,
    [switch]$Smoke
)

$ErrorActionPreference = 'Continue'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$TrainScript = Join-Path $RepoRoot 'scripts\train_teacher.py'

# Curated training manifest (HQ train)
$Manifest = Join-Path $RepoRoot 'Training_data_cleaned\classification_manifest_hq_train.csv'
$OutRoot = Join-Path $RepoRoot 'Training_data_cleaned'

if (!(Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
if (!(Test-Path $TrainScript)) { throw "Script not found: $TrainScript" }
if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogDir = Join-Path $RepoRoot (Join-Path 'outputs\teachers' (Join-Path '_overnight_logs' ("rn18_b3_" + $Stamp)))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$models = @(
    @{ Tag = 'RN18'; Name = 'resnet18' },
    @{ Tag = 'B3';   Name = 'tf_efficientnet_b3' }
)

Write-Host "Logs: $LogDir"
Write-Host "Stage=$Stage Epochs=$Epochs BatchSize=$BatchSize NumWorkers=$NumWorkers EvalEvery=$EvalEvery Seed=$Seed"

function Invoke-TeacherRun {
    param(
        [string]$Tag,
        [string]$Name,
        [string]$StageTag,
        [int]$ImageSize,
        [string[]]$ExtraArgs,
        [string]$ResumePath
    )

    $logPath = Join-Path $LogDir ("$Tag`_$Name`_$StageTag.log")
    $outDir = Join-Path $RepoRoot ("outputs\teachers\" + $Tag + "_" + $Name + "_" + $StageTag + "_seed" + $Seed)

    $args = @(
        $TrainScript,
        '--model', $Name,
        '--manifest', $Manifest,
        '--out-root', $OutRoot,
        '--image-size', $ImageSize,
        '--batch-size', $BatchSize,
        '--num-workers', $NumWorkers,
        '--seed', $Seed,
        '--accum-steps', $AccumSteps,
        '--eval-every', $EvalEvery,
        '--max-epochs', $Epochs,
        '--checkpoint-every', 10,
        '--output-dir', $outDir
    )

    if ($Clahe) { $args += @('--clahe') }
    if ($SkipOnnxDuringTrain) { $args += @('--skip-onnx-during-train') }
    if ($Smoke) { $args += @('--smoke') }

    if ($ResumePath -and (Test-Path $ResumePath)) {
        $args += @('--resume', $ResumePath)
    }

    if ($ExtraArgs -and $ExtraArgs.Length -gt 0) {
        $args += $ExtraArgs
    }

    Write-Host "\n=== [$Tag] $Name ($StageTag) ==="
    Write-Host "Output: $outDir"
    Write-Host "Logging to: $logPath"

    & $PythonExe @args 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "Teacher run failed for $Name ($StageTag), exit code $LASTEXITCODE. See log: $logPath"
    }

    return $outDir
}

foreach ($m in $models) {
    $tag = $m.Tag
    $name = $m.Name

    $stageAOut = $null

    if ($Stage -eq 'A' -or $Stage -eq 'AB') {
        # Stage A: 224 pretrain INCLUDING ferplus (no source filter)
        $stageAOut = Invoke-TeacherRun -Tag $tag -Name $name -StageTag 'stageA_img224' -ImageSize 224 -ExtraArgs @() -ResumePath ''
    }

    if ($Stage -eq 'B' -or $Stage -eq 'AB') {
        # Stage B: 384 finetune EXCLUDING ferplus
        # Resume from Stage A checkpoint_last if Stage A ran, else try to auto-resume from output-dir checkpoint_last.
        $resume = ''
        if ($stageAOut) {
            $resumeCandidate = Join-Path $stageAOut 'checkpoint_last.pt'
            if (Test-Path $resumeCandidate) { $resume = $resumeCandidate }
        }

        $extra = @('--exclude-sources', 'ferplus')
        [void](Invoke-TeacherRun -Tag $tag -Name $name -StageTag 'stageB_img384_noferplus' -ImageSize 384 -ExtraArgs $extra -ResumePath $resume)
    }
}

Write-Host "\nAll selected teacher runs completed."
