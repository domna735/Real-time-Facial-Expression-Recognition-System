param(
    [string]$RepoRoot = "",

    # Baseline KD run folder used only to locate the resume checkpoint (checkpoint_last.pt) by default.
    [string]$BaselineKD = "outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119",

    # Baseline DKD run folder to compare against (same student/data setup).
    [string]$BaselineDKD = "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722",

    # Common runtime
    [int]$NLBatchSize = 64,
    [int]$NegLBatchSize = 128,
    [int]$NumWorkers = 8,
    [int]$DkdEpochs = 5,

    # Optional consistency with prior runs
    [switch]$UseClahe,
    [switch]$UseAmp,

    # Resume checkpoint for DKD stage.
    # If empty, defaults to <BaselineKD>/checkpoint_last.pt
    [string]$DkdResumeFrom = "",

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

    # Optional: pass sweep points as a single CSV string (e.g., "0.3,0.5").
    # This is more robust when this script is invoked from another PowerShell script.
    [string]$NegLEntropyThreshesCsv = "",

    # Planned synergy run (NL + NegL together)
    [double]$SynergyNegLEntropyThresh = 0.4,

    # Repro
    [string]$Model = "mobilenetv3_large_100",
    [int]$ImageSize = 224,
    [int]$Seed = 1337
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

function Resolve-PythonExe {
    param([string]$Root)
    $py = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $py) { return $py }
    $pyAlt = Join-Path $Root ".venv\Scripts\python"
    if (Test-Path -LiteralPath $pyAlt) { return $pyAlt }
    return "python"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

$RepoRoot = Resolve-RepoRoot -Provided $RepoRoot
$PythonExe = Resolve-PythonExe -Root $RepoRoot

$runner = Join-Path $RepoRoot "scripts\run_student_mnv3_ce_kd_dkd.ps1"
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Runner not found: $runner"
}

$baselineKdAbs = $BaselineKD
if (-not [System.IO.Path]::IsPathRooted($baselineKdAbs)) {
    $baselineKdAbs = Join-Path $RepoRoot $BaselineKD
}
if (-not (Test-Path -LiteralPath $baselineKdAbs)) {
    throw "BaselineKD not found: $baselineKdAbs"
}

$baselineDkdAbs = $BaselineDKD
if (-not [System.IO.Path]::IsPathRooted($baselineDkdAbs)) {
    $baselineDkdAbs = Join-Path $RepoRoot $BaselineDKD
}
if (-not (Test-Path -LiteralPath $baselineDkdAbs)) {
    throw "BaselineDKD not found: $baselineDkdAbs"
}

if (-not $DkdResumeFrom) {
    $DkdResumeFrom = Join-Path $baselineKdAbs "checkpoint_last.pt"
}
$dkdResumeAbs = $DkdResumeFrom
if (-not [System.IO.Path]::IsPathRooted($dkdResumeAbs)) {
    $dkdResumeAbs = Join-Path $RepoRoot $DkdResumeFrom
}
if (-not (Test-Path -LiteralPath $dkdResumeAbs)) {
    throw "DkdResumeFrom checkpoint not found: $dkdResumeAbs"
}

function Parse-DoubleList {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [string]$Name = "values"
    )

    $items = $Text -split "[\s,;]+" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
    if ($items.Count -eq 0) {
        throw "${Name}: no values found in '$Text'"
    }

    $out = New-Object System.Collections.Generic.List[double]
    foreach ($s in $items) {
        $v = $null
        if (-not [double]::TryParse($s, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$v)) {
            throw "${Name}: could not parse '$s' as double (from '$Text')"
        }
        $out.Add($v)
    }
    return ,$out.ToArray()
}

if ($NegLEntropyThreshesCsv) {
    $NegLEntropyThreshes = Parse-DoubleList -Text $NegLEntropyThreshesCsv -Name "NegLEntropyThreshesCsv"
}

$stampOuter = Get-Date -Format yyyyMMdd_HHmmss
$compareDir = Join-Path $RepoRoot "outputs\students"
Ensure-Dir $compareDir

function Invoke-DKDStep {
    param(
        [string]$Name,
        [int]$BatchSize,
        [string[]]$StepArgs,
        [string]$CompareOutName
    )

    Write-Host "\n==============================="
    Write-Host "DKD RUN: $Name"
    Write-Host "==============================="

    $common = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $runner,
        "-SkipSmoke",
        "-CeEpochs", 0,
        "-KdEpochs", 0,
        "-DkdEpochs", $DkdEpochs,
        "-BatchSize", $BatchSize,
        "-NumWorkers", $NumWorkers,
        "-DkdResumeFrom", $dkdResumeAbs,
        "-Model", $Model,
        "-ImageSize", $ImageSize,
        "-Seed", $Seed
    )

    if ($UseClahe) { $common += "-UseClahe" }
    if ($UseAmp) { $common += "-UseAmp" }

    $cmd = @("powershell") + $common + $StepArgs
    Write-Host ("CMD: " + ($cmd -join " "))

    $startedAt = Get-Date
    $outputText = & powershell @common @StepArgs 2>&1 | Tee-Object -Variable tee
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit $LASTEXITCODE)"
    }

    # Locate the newest DKD output directory. We cannot reliably parse "Run stamp" because
    # the runner prints it via Write-Host (host stream), which doesn't flow through pipes.
    $dkdRoot = Join-Path $RepoRoot "outputs\students\DKD"
    if (-not (Test-Path -LiteralPath $dkdRoot)) {
        throw "DKD outputs root not found: $dkdRoot"
    }
    $pattern = "${Model}_img${ImageSize}_seed${Seed}_DKD_*"
    $dkdCand = Get-ChildItem -LiteralPath $dkdRoot -Directory |
        Where-Object { $_.Name -like $pattern } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $dkdCand) {
        throw "No DKD output dirs found under: $dkdRoot (pattern=$pattern)"
    }
    if ($dkdCand.LastWriteTime -lt $startedAt.AddMinutes(-10)) {
        Write-Host "[warn] Newest DKD dir looks old (${($dkdCand.LastWriteTime)}); continuing anyway: ${($dkdCand.FullName)}"
    }
    $dkdOut = $dkdCand.FullName

    $compareOutAbs = Join-Path $compareDir $CompareOutName
    Write-Host "Comparing:"
    Write-Host "  baseline DKD: $baselineDkdAbs"
    Write-Host "  new DKD     : $dkdOut"
    Write-Host "  out         : $compareOutAbs"

    & $PythonExe tools/diagnostics/compare_student_runs.py `
        $baselineDkdAbs $dkdOut `
        --label "Baseline DKD (reference)" `
        --label "New DKD (this stamp)" `
        --out $compareOutAbs

    if ($LASTEXITCODE -ne 0) {
        throw "compare_student_runs.py failed for step: $Name"
    }

    Write-Host "Compare markdown: $compareOutAbs"
}

# 1) NL-only (top-k gate)
Invoke-DKDStep -Name "NL-only: NL(proto, penultimate) gate=topk frac=$NLTopKFrac w=$NLWeight" `
    -BatchSize $NLBatchSize `
    -CompareOutName ("_compare_${stampOuter}_dkd${DkdEpochs}_nlproto_penultimate_topk" + ($NLTopKFrac.ToString().Replace('.','p')) + "_w" + ($NLWeight.ToString().Replace('.','p')) + "_vs_dkd.md") `
    -StepArgs @(
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
    Invoke-DKDStep -Name "NegL-only: gate=$NegLGate ent=$thr w=$NegLWeight ratio=$NegLRatio" `
        -BatchSize $NegLBatchSize `
        -CompareOutName ("_compare_${stampOuter}_dkd${DkdEpochs}_negl_${NegLGate}_ent" + ($thr.ToString().Replace('.','p')) + "_vs_dkd.md") `
        -StepArgs @(
            "-UseNegL",
            "-NegLWeight", $NegLWeight,
            "-NegLRatio", $NegLRatio,
            "-NegLGate", $NegLGate,
            "-NegLEntropyThresh", $thr
        )
}

# 3) Synergy run (NL + NegL together)
Invoke-DKDStep -Name "Synergy: NL(topk=$NLTopKFrac) + NegL($NegLGate ent=$SynergyNegLEntropyThresh)" `
    -BatchSize $NegLBatchSize `
    -CompareOutName ("_compare_${stampOuter}_dkd${DkdEpochs}_nlproto_topk" + ($NLTopKFrac.ToString().Replace('.','p')) + "_plus_negl_${NegLGate}_ent" + ($SynergyNegLEntropyThresh.ToString().Replace('.','p')) + "_vs_dkd.md") `
    -StepArgs @(
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

Write-Host "\nAll DKD next-planned runs completed. Compare markdowns are under: $compareDir"