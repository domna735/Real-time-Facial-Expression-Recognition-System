param(
    [string]$Manifest = "Training_data_cleaned/classification_manifest_hq_train.csv",
    [string]$OutRoot = "Training_data_cleaned",
    [int]$StageAImageSize = 224,
    [int]$StageBImageSize = 384,
    [int]$NumWorkers = 0,
    [int]$MaxTrainBatches = 10,
    [int]$MaxValBatches = 5,
    [switch]$Clean,
    [switch]$NoClahe
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    $here = Split-Path -Parent $PSCommandPath
    return (Resolve-Path (Join-Path $here ".."))
}

function Get-PythonExe([string]$repoRoot) {
    $venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        return $venvPy
    }
    return "python"
}

function Require-File([string]$path) {
    if (!(Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

$repoRoot = (Resolve-RepoRoot).Path
$py = Get-PythonExe $repoRoot
$trainScript = Join-Path $repoRoot "scripts\train_teacher.py"

$includeStageA = "ferplus,rafdb_basic,affectnet_full_balanced,expw_hq"
$excludeStageB = "ferplus"

$models = @(
    @{ name = "resnet18"; bsA = 32; bsB = 16 },
    @{ name = "tf_efficientnet_b3"; bsA = 16; bsB = 14 }
)

Write-Host "Repo:  $repoRoot"
Write-Host "Python: $py"
Write-Host "Manifest: $Manifest"
Write-Host "Smoke Stage A->B for: resnet18 + tf_efficientnet_b3"

foreach ($m in $models) {
    $model = $m.name
    $bsA = [int]$m.bsA
    $bsB = [int]$m.bsB

    # BalancedBatchSampler requires batch_size >= num_classes * min_per_class.
    $minBatch = 14  # 7 classes * min_per_class(2)
    if ($bsA -lt $minBatch) { $bsA = $minBatch }
    if ($bsB -lt $minBatch) { $bsB = $minBatch }

    $tag = ($model -replace "[^a-zA-Z0-9_]+", "_")
    $outA = Join-Path $repoRoot ("outputs\teachers\_smoke_stageA_{0}_img{1}" -f $tag, $StageAImageSize)
    $outB = Join-Path $repoRoot ("outputs\teachers\_smoke_stageB_{0}_img{1}" -f $tag, $StageBImageSize)

    if ($Clean) {
        Remove-Item -Recurse -Force $outA, $outB -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "=== [$model] Stage A (img=$StageAImageSize, include=$includeStageA) ==="
    $argsA = @(
        $trainScript,
        "--model", $model,
        "--manifest", $Manifest,
        "--out-root", $OutRoot,
        "--image-size", "$StageAImageSize",
        "--batch-size", "$bsA",
        "--num-workers", "$NumWorkers",
        "--max-train-batches", "$MaxTrainBatches",
        "--max-val-batches", "$MaxValBatches",
        "--eval-every", "1",
        "--max-epochs", "1",
        "--checkpoint-every", "1",
        "--output-dir", $outA,
        "--include-sources", $includeStageA,
        "--skip-onnx-during-train"
    )
    if (-not $NoClahe) { $argsA += "--clahe" }

    & $py @argsA
    if ($LASTEXITCODE -ne 0) { throw "Stage A smoke failed for $model" }

    $ckptA = Join-Path $outA "checkpoint_last.pt"
    Require-File $ckptA

    Write-Host ""
    Write-Host "=== [$model] Stage B (img=$StageBImageSize, exclude=$excludeStageB; init-from Stage A) ==="
    $argsB = @(
        $trainScript,
        "--model", $model,
        "--manifest", $Manifest,
        "--out-root", $OutRoot,
        "--image-size", "$StageBImageSize",
        "--batch-size", "$bsB",
        "--num-workers", "$NumWorkers",
        "--max-train-batches", "$MaxTrainBatches",
        "--max-val-batches", "$MaxValBatches",
        "--eval-every", "1",
        "--max-epochs", "1",
        "--checkpoint-every", "1",
        "--output-dir", $outB,
        "--exclude-sources", $excludeStageB,
        "--init-from", $ckptA,
        "--skip-onnx-during-train"
    )
    if (-not $NoClahe) { $argsB += "--clahe" }

    & $py @argsB
    if ($LASTEXITCODE -ne 0) { throw "Stage B smoke failed for $model" }

    # Validate artifacts (Stage A and Stage B)
    foreach ($dir in @($outA, $outB)) {
        Require-File (Join-Path $dir "alignmentreport.json")
        Require-File (Join-Path $dir "history.json")
        Require-File (Join-Path $dir "reliabilitymetrics.json")
        Require-File (Join-Path $dir "calibration.json")
        Require-File (Join-Path $dir "checkpoint_last.pt")
        Require-File (Join-Path $dir "last.onnx")
    }

    Write-Host "OK: $model Stage A->B smoke + artifacts" -ForegroundColor Green
}

Write-Host "\nAll smoke tests passed (RN18 + B3, Stage A->B)." -ForegroundColor Green
