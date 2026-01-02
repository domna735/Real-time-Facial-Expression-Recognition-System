param(
    [int]$ImageSize = 384,
    [int]$BatchSize = 64,
    [int]$Epochs = 60,
    [int]$NumWorkers = 4,
    [int]$Seed = 1337,
    [int]$AccumSteps = 1,
    [int]$EvalEvery = 1,
    [switch]$NoClahe,
    [switch]$SkipOnnxDuringTrain,
    [switch]$Smoke
)

$ErrorActionPreference = 'Continue'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$TrainScript = Join-Path $RepoRoot 'scripts\train_teacher.py'
$Manifest = Join-Path $RepoRoot 'Training_data_cleaned\classification_manifest.csv'
$OutRoot = Join-Path $RepoRoot 'Training_data_cleaned'

if (!(Test-Path $PythonExe)) { throw "Python not found: $PythonExe" }
if (!(Test-Path $TrainScript)) { throw "Script not found: $TrainScript" }
if (!(Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogDir = Join-Path $RepoRoot (Join-Path 'outputs\teachers' (Join-Path '_overnight_logs' $Stamp))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$models = @(
    @{ Tag = 'RN18'; Name = 'resnet18' },
    @{ Tag = 'CNXT'; Name = 'convnext_tiny' }
)

# To run additional teachers later, add them back like:
#   @{ Tag = 'RN50'; Name = 'resnet50' },
#   @{ Tag = 'B3';   Name = 'tf_efficientnet_b3' },
#   @{ Tag = 'ViT';  Name = 'vit_base_patch16_384' },

Write-Host "Logs: $LogDir"
Write-Host "ImageSize=$ImageSize BatchSize=$BatchSize Epochs=$Epochs Smoke=$Smoke"

foreach ($m in $models) {
    $tag = $m.Tag
    $name = $m.Name
    $logPath = Join-Path $LogDir "$tag`_$name.log"

    $outDir = Join-Path $RepoRoot ("outputs\teachers\" + $tag + "_" + $name + "_img" + $ImageSize + "_seed" + $Seed)

    $args = @(
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
        '--checkpoint-every', 10,
        '--output-dir', $outDir
    )

    if (-not $NoClahe) {
        $args += @('--clahe')
    }

    if ($SkipOnnxDuringTrain) {
        $args += @('--skip-onnx-during-train')
    }

    if ($Smoke) {
        $args += @('--smoke')
    }

    Write-Host "\n=== [$tag] $name ==="
    Write-Host "Logging to: $logPath"

    & $PythonExe @args 2>&1 | Tee-Object -FilePath $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "Teacher run failed for $name (exit code $LASTEXITCODE). See log: $logPath"
    }
}

Write-Host "\nAll teacher runs completed."