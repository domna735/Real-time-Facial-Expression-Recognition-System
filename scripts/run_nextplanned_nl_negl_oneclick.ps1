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

    # NL knobs
    [double]$NLWeight = 0.1,
    [int]$NLDim = 32,
    [double]$NLMomentum = 0.9,

    # Planned NL run: top-k gate (force sustained activity)
    [double]$NLTopKFrac = 0.05,

    # Planned NegL sweep points (entropy-gated)
    [double]$NegLWeight = 0.05,
    [double]$NegLRatio = 0.5,
    [ValidateSet("none","entropy")][string]$NegLGate = "entropy",
    [double[]]$NegLEntropyThreshes = @(0.3, 0.5),

    # Planned synergy run (NL + NegL together)
    [double]$SynergyNegLEntropyThresh = 0.4,

    # Optional: also run the DKD version of this planned set (resume-from-KD).
    [switch]$AlsoRunDKD,

    # DKD baseline and schedule (only used when -AlsoRunDKD)
    [string]$BaselineDKD = "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722",
    [int]$DkdEpochs = 5,

    # Optional: override DKD resume checkpoint (only used when -AlsoRunDKD).
    # If empty, DKD script defaults to <BaselineKD>/checkpoint_last.pt
    [string]$DkdResumeFrom = ""
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

# 1) NL-only (top-k gate; sustained applied_frac)
Invoke-Step -Name "NL-only: NL(proto, penultimate) gate=topk frac=$NLTopKFrac w=$NLWeight" `
    -CompareOutName ("_compare_${stamp}_kd${KdEpochs}_nlproto_penultimate_topk" + ($NLTopKFrac.ToString().Replace('.','p')) + "_w" + ($NLWeight.ToString().Replace('.','p')) + "_vs_kd.md") `
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

# 2) NegL-only sweep points
foreach ($thr in $NegLEntropyThreshes) {
    Invoke-Step -Name "NegL-only: gate=$NegLGate ent=$thr w=$NegLWeight ratio=$NegLRatio" `
        -CompareOutName ("_compare_${stamp}_kd${KdEpochs}_negl_${NegLGate}_ent" + ($thr.ToString().Replace('.','p')) + "_vs_kd.md") `
        -StepArgs @(
            "-BatchSize", $NegLBatchSize,
            "-UseNegL",
            "-NegLWeight", $NegLWeight,
            "-NegLRatio", $NegLRatio,
            "-NegLGate", $NegLGate,
            "-NegLEntropyThresh", $thr
        )
}

# 3) Synergy run (NL + NegL together)
Invoke-Step -Name "Synergy: NL(topk=$NLTopKFrac) + NegL($NegLGate ent=$SynergyNegLEntropyThresh)" `
    -CompareOutName ("_compare_${stamp}_kd${KdEpochs}_nlproto_topk" + ($NLTopKFrac.ToString().Replace('.','p')) + "_plus_negl_${NegLGate}_ent" + ($SynergyNegLEntropyThresh.ToString().Replace('.','p')) + "_vs_kd.md") `
    -StepArgs @(
        "-BatchSize", $NegLBatchSize,
        "-UseNL",
        "-NLKind", "proto",
        "-NLEmbed", "penultimate",
        "-NLDim", $NLDim,
        "-NLMomentum", $NLMomentum,
        "-NLProtoGate", "topk",
        "-NLTopKFrac", $NLTopKFrac,
        "-NLWeight", $NLWeight,
        "-UseNegL",
        "-NegLWeight", $NegLWeight,
        "-NegLRatio", $NegLRatio,
        "-NegLGate", $NegLGate,
        "-NegLEntropyThresh", $SynergyNegLEntropyThresh
    )

Write-Host "\nAll next-planned runs completed. Compare markdowns are under: $compareDir"

if ($AlsoRunDKD) {
    $dkdScript = Join-Path $RepoRoot "scripts\run_nextplanned_nl_negl_oneclick_dkd.ps1"
    if (-not (Test-Path -LiteralPath $dkdScript)) {
        throw "DKD one-click script not found: $dkdScript"
    }

    $dkdArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $dkdScript,
        "-RepoRoot", $RepoRoot,
        "-BaselineKD", $BaselineKD,
        "-BaselineDKD", $BaselineDKD,
        "-NLBatchSize", $NLBatchSize,
        "-NegLBatchSize", $NegLBatchSize,
        "-NumWorkers", $NumWorkers,
        "-DkdEpochs", $DkdEpochs,
        "-NLWeight", $NLWeight,
        "-NLDim", $NLDim,
        "-NLMomentum", $NLMomentum,
        "-NLTopKFrac", $NLTopKFrac,
        "-NegLWeight", $NegLWeight,
        "-NegLRatio", $NegLRatio,
        "-NegLGate", $NegLGate,
        "-SynergyNegLEntropyThresh", $SynergyNegLEntropyThresh
    )

    # Pass the entropy sweep thresholds as a single CSV string.
    # This avoids occasional argument mis-binding when invoking a nested powershell.exe.
    $dkdArgs += @("-NegLEntropyThreshesCsv", ($NegLEntropyThreshes -join ","))

    if ($UseClahe) { $dkdArgs += "-UseClahe" }
    if ($UseAmp) { $dkdArgs += "-UseAmp" }
    if ($DkdResumeFrom) { $dkdArgs += @("-DkdResumeFrom", $DkdResumeFrom) }

    Write-Host "\n==============================="
    Write-Host "ALSO RUN DKD: invoking DKD one-click"
    Write-Host "==============================="
    Write-Host ("CMD: powershell " + ($dkdArgs -join " "))

    & powershell @dkdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "DKD one-click script failed (exit $LASTEXITCODE)"
    }
}
