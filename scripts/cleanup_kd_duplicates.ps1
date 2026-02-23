param(
    [string]$RepoRoot = "",

    # Where KD student runs live.
    [string]$KDDir = "outputs/students/KD",

    # One fixed baseline to always keep.
    [string]$KeepBaseline = "outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119",

    # Also keep any KD dirs referenced inside these markdown files.
    [string[]]$KeepFromMd = @(
        "research/nl_negl_plan/NL_NGEL_study.md",
        "research/process_log/Jan process log/Jan_week1_process_log.md"
    ),

    # Where to move redundant runs.
    [string]$RedundantDir = "outputs/students/KD/_redundant",

    # Within each duplicate group, which run to keep.
    [ValidateSet("newest","oldest","best")][string]$Prefer = "newest",

    # How strict to be when deciding duplicates.
    # - knobs: same research knobs (ignores workers, batch size, etc.)
    # - full : stricter (includes runtime knobs)
    [ValidateSet("knobs","full")][string]$SignatureLevel = "knobs",

    [switch]$DryRun
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

$KDAbs = $KDDir
if (-not [System.IO.Path]::IsPathRooted($KDAbs)) { $KDAbs = Join-Path $RepoRoot $KDDir }
if (-not (Test-Path -LiteralPath $KDAbs)) { throw "KDDir not found: $KDAbs" }

$RedundantAbs = $RedundantDir
if (-not [System.IO.Path]::IsPathRooted($RedundantAbs)) { $RedundantAbs = Join-Path $RepoRoot $RedundantDir }
Ensure-Dir $RedundantAbs

$KeepBaselineAbs = $KeepBaseline
if (-not [System.IO.Path]::IsPathRooted($KeepBaselineAbs)) { $KeepBaselineAbs = Join-Path $RepoRoot $KeepBaseline }
if (-not (Test-Path -LiteralPath $KeepBaselineAbs)) { throw "KeepBaseline not found: $KeepBaselineAbs" }

$helper = Join-Path $RepoRoot "tools\diagnostics\find_duplicate_kd_runs.py"
if (-not (Test-Path -LiteralPath $helper)) { throw "Helper not found: $helper" }

$argsList = @(
    $helper,
    "--repo-root", $RepoRoot,
    "--kd-root", $KDAbs,
    "--prefer", $Prefer,
    "--signature-level", $SignatureLevel,
    "--keep", $KeepBaselineAbs
)

foreach ($md in $KeepFromMd) {
    $mdAbs = $md
    if (-not [System.IO.Path]::IsPathRooted($mdAbs)) { $mdAbs = Join-Path $RepoRoot $md }
    if (Test-Path -LiteralPath $mdAbs) {
        $argsList += @("--keep-from-md", $mdAbs)
    }
}

Write-Host "Computing duplicate KD runs..."
$json = & $PythonExe @argsList | Out-String
if ($LASTEXITCODE -ne 0) { throw "find_duplicate_kd_runs.py failed (exit $LASTEXITCODE)" }

$data = $json | ConvertFrom-Json
$toMove = @($data.move)

Write-Host "KD root        : $($data.kd_root)"
Write-Host "Prefer keep    : $Prefer"
Write-Host "SignatureLevel : $SignatureLevel"
Write-Host "Keep baseline  : $KeepBaselineAbs"
Write-Host "Redundant dir  : $RedundantAbs"
Write-Host "Duplicates     : $($toMove.Count) folders to move"

if ($toMove.Count -eq 0) {
    Write-Host "No redundant KD folders found."
    exit 0
}

# Write a small plan file for traceability.
$planPath = Join-Path $RedundantAbs ("_move_plan_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".txt")
$toMove | Set-Content -LiteralPath $planPath -Encoding UTF8
Write-Host "Move plan saved: $planPath"

if ($DryRun) {
    Write-Host "DryRun: not moving anything. First 20 candidates:"
    $toMove | Select-Object -First 20 | ForEach-Object { Write-Host "  $_" }
    exit 0
}

# Move folders.
foreach ($src in $toMove) {
    $srcPath = [string]$src
    if (-not (Test-Path -LiteralPath $srcPath)) {
        Write-Host "[skip] missing: $srcPath"
        continue
    }

    $name = Split-Path -Leaf $srcPath
    $dst = Join-Path $RedundantAbs $name

    # Avoid collisions
    if (Test-Path -LiteralPath $dst) {
        $i = 1
        while (Test-Path -LiteralPath ($dst + "_dup" + $i)) { $i++ }
        $dst = $dst + "_dup" + $i
    }

    Write-Host "Moving: $srcPath"
    Write-Host "   ->  $dst"
    Move-Item -LiteralPath $srcPath -Destination $dst
}

Write-Host "Done. Redundant KD runs are under: $RedundantAbs"
