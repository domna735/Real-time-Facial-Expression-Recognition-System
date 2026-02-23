param(
    [string]$RepoRoot = "",

    # Baseline KD run folder to compare against (for the compare markdown outputs).
    # Default points to your existing KD-only 5ep baseline.
    [string]$BaselineKD = "outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119",

    # Common runtime
    [int]$NLBatchSize = 64,
    [int]$NegLBatchSize = 128,
    [int]$NumWorkers = 8,
    [int]$KdEpochs = 5,

    # Optional consistency with prior runs
    [switch]$UseClahe,
    [switch]$UseAmp,

    # NL knobs for the two NL runs
    [double]$NLWeight = 0.1,
    [int]$NLDim = 32,
    [double]$NLMomentum = 0.9,

    # Run 1: fixed threshold
    [double]$NLFixedThresh = 0.05,

    # Run 2: top-k gate
    [double]$NLTopKFrac = 0.10,

    # Run 3: NegL-only gate sweep point
    [double]$NegLWeight = 0.05,
    [double]$NegLRatio = 0.5,
    [ValidateSet("none","entropy")][string]$NegLGate = "entropy",
    [double]$NegLEntropyThresh = 0.4
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$Provided)
    if ($Provided -and (Test-Path -LiteralPath $Provided)) {
        return (Resolve-Path -LiteralPath $Provided).Path
    }
    $here = Split-Path -Parent $PSCommandPath
    $root = Resolve-Path -LiteralPath (Join-Path $here "..");
    return $root.Path
}

$RepoRoot = Resolve-RepoRoot -Provided $RepoRoot
$runner = Join-Path $RepoRoot "scripts\run_student_mnv3_ce_kd_dkd.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner not found: $runner"
}

$baselineAbs = $BaselineKD
if (-not [System.IO.Path]::IsPathRooted($baselineAbs)) {
    $baselineAbs = Join-Path $RepoRoot $BaselineKD
}
if (-not (Test-Path -LiteralPath $baselineAbs)) {
    throw "BaselineKD not found: $baselineAbs"
}

$stamp = Get-Date -Format yyyyMMdd_HHmmss
$compareDir = Join-Path $RepoRoot "outputs\students"

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$StepArgs,
        [string]$CompareOutName
    )

    Write-Host "\n==============================="
    Write-Host "RUN: $Name"
    Write-Host "==============================="

    $compareOut = Join-Path $compareDir $CompareOutName

    $common = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $runner,
        "-SkipSmoke",
        "-CeEpochs", 0,
        "-KdEpochs", $KdEpochs,
        "-DkdEpochs", 0,
        "-NumWorkers", $NumWorkers,
        "-CompareWith", $baselineAbs,
        "-CompareOut", $compareOut
    )

    if ($UseClahe) { $common += "-UseClahe" }
    if ($UseAmp) { $common += "-UseAmp" }

    $cmd = @("powershell") + $common + $StepArgs
    Write-Host ("CMD: " + ($cmd -join " "))

    & powershell @common @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit $LASTEXITCODE)"
    }

    Write-Host "Compare markdown: $compareOut"
}

# 1) NL(proto, penultimate) fixed threshold (try to sustain activity by lowering thr)
Invoke-Step -Name "NL(proto, penultimate) gate=fixed thr=$NLFixedThresh" `
    -CompareOutName ("_compare_${stamp}_kd5_nlproto_penultimate_fixed_thr" + ($NLFixedThresh.ToString().Replace('.','p')) + "_vs_kd5.md") `
    -StepArgs @(
        "-BatchSize", $NLBatchSize,
        "-UseNL",
        "-NLKind", "proto",
        "-NLEmbed", "penultimate",
        "-NLDim", $NLDim,
        "-NLMomentum", $NLMomentum,
        "-NLProtoGate", "fixed",
        "-NLConsistencyThresh", $NLFixedThresh,
        "-NLWeight", $NLWeight
    )

# 2) NL(proto, penultimate) top-k gating (forces applied_frac ~= NLTopKFrac)
Invoke-Step -Name "NL(proto, penultimate) gate=topk frac=$NLTopKFrac" `
    -CompareOutName ("_compare_${stamp}_kd5_nlproto_penultimate_topk" + ($NLTopKFrac.ToString().Replace('.','p')) + "_vs_kd5.md") `
    -StepArgs @(
        "-BatchSize", $NLBatchSize,
        "-UseNL",
        "-NLKind", "proto",
        "-NLEmbed", "penultimate",
        "-NLDim", $NLDim,
        "-NLMomentum", $NLMomentum,
        "-NLProtoGate", "topk",
        "-NLTopKFrac", $NLTopKFrac,
        "-NLWeight", $NLWeight
    )

# 3) NegL-only (make it bite)
Invoke-Step -Name "NegL-only gate=$NegLGate ent=$NegLEntropyThresh w=$NegLWeight ratio=$NegLRatio" `
    -CompareOutName ("_compare_${stamp}_kd5_negl_${NegLGate}_ent" + ($NegLEntropyThresh.ToString().Replace('.','p')) + "_vs_kd5.md") `
    -StepArgs @(
        "-BatchSize", $NegLBatchSize,
        "-UseNegL",
        "-NegLWeight", $NegLWeight,
        "-NegLRatio", $NegLRatio,
        "-NegLGate", $NegLGate,
        "-NegLEntropyThresh", $NegLEntropyThresh
    )

Write-Host "\nAll 3 runs completed. Compare markdowns are under: $compareDir"
