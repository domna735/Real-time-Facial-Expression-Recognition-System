param(
  [string]$Manifest = "Training_data_cleaned\\classification_manifest_eval_only.csv",
  [string]$Split = "test",
  [int]$ImageSize = 224,
  [string]$OutRoot = "outputs\\softlabels",
  [int]$CudaDevice = 0,
  [int]$BatchSize = 128,
  [int]$NumWorkers = 0,
  [switch]$UseClahe
)

$ErrorActionPreference = "Stop"

$py = "C:/Real-time-Facial-Expression-Recognition-System_v2_restart/.venv/Scripts/python.exe"

$rn18 = "outputs\\teachers\\RN18_resnet18_seed1337_stageA_img224\\best.pt"
$b3   = "outputs\\teachers\\B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224\\best.pt"

$tag = Get-Date -Format "yyyyMMdd_HHmmss"

$cases = @(
  @{ name = "rn18_0p3_b3_0p7"; wa = 0.3; wb = 0.7 },
  @{ name = "rn18_0p5_b3_0p5"; wa = 0.5; wb = 0.5 },
  @{ name = "rn18_0p7_b3_0p3"; wa = 0.7; wb = 0.3 }
)

foreach ($c in $cases) {
  $outDir = Join-Path $OutRoot ("ensemble_stageA_{0}_{1}" -f $c.name, $tag)

  Write-Host "=== Export ensemble: $($c.name) -> $outDir ==="

  $args = @(
    "scripts\\export_ensemble_softlabels.py",
    "--manifest", $Manifest,
    "--split", $Split,
    "--image-size", $ImageSize,
    "--teacher-a", $rn18,
    "--teacher-b", $b3,
    "--weight-a", $c.wa,
    "--weight-b", $c.wb,
    "--out-root", $outDir,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--device", "cuda"
  )

  if ($UseClahe) {
    $args += "--use-clahe"
  }

  $env:CUDA_VISIBLE_DEVICES = "$CudaDevice"
  & $py @args

  Write-Host "--- Diagnose alignment: $outDir ---"
  & $py scripts\\diagnose_alignment.py --manifest $Manifest --softlabels-dir $outDir --require-classorder
}
