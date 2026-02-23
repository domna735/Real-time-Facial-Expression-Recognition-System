param(
    [string]$RepoRoot = "",

    # Root containing KD/DKD folders.
    [string]$StudentsRoot = "outputs/students",

    # Where to place archives.
    [string]$ArchiveRoot = "outputs/students/_archive",

    # Baselines that must stay in place for one-click scripts.
    [string]$KeepBaselineKD = "outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_182119",
    [string]$KeepBaselineDKD = "outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722",

    # Markdown logs to scan for referenced run dirs to keep.
    [string[]]$KeepFromMd = @(
        "research/nl_negl_plan/NL_NGEL_study.md",
        "research/process_log/Jan process log/Jan_week1_process_log.md"
    ),

    # Options
    [switch]$DryRun,
    [switch]$Zip,
    [switch]$RemoveAfterZip
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

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Normalize-Path {
    param([string]$P)
    if (-not $P) { return $null }
    $q = $P
    if (-not [System.IO.Path]::IsPathRooted($q)) {
        $q = Join-Path $RepoRoot $q
    }
    return (Resolve-Path -LiteralPath $q).Path
}

function Get-RunDateKey {
    param(
        [string]$RunName
    )
    # Expect ..._KD_YYYYMMDD_HHMMSS or ..._DKD_YYYYMMDD_HHMMSS
    $m = [regex]::Match($RunName, "_(KD|DKD)_(\d{8})_(\d{6})$")
    if (-not $m.Success) {
        return "unknown"
    }
    $d = $m.Groups[2].Value
    return ($d.Substring(0,4) + "-" + $d.Substring(4,2) + "-" + $d.Substring(6,2))
}

function Extract-KeepNamesFromMarkdown {
    param([string[]]$MdPaths)

    $keep = New-Object System.Collections.Generic.HashSet[string]
    $pat = [regex]"outputs/students/(KD|DKD)/([A-Za-z0-9_\-]+)"

    foreach ($mp in $MdPaths) {
        if (-not (Test-Path -LiteralPath $mp)) { continue }
        $text = Get-Content -LiteralPath $mp -Raw -ErrorAction SilentlyContinue
        if (-not $text) { continue }
        $norm = $text -replace "\\\\", "/"
        foreach ($m in $pat.Matches($norm)) {
            [void]$keep.Add($m.Groups[2].Value)
        }
    }

    return ,$keep
}

$RepoRoot = Resolve-RepoRoot -Provided $RepoRoot

$studentsAbs = Normalize-Path -P $StudentsRoot
$archiveAbs = $ArchiveRoot
if (-not [System.IO.Path]::IsPathRooted($archiveAbs)) {
    $archiveAbs = Join-Path $RepoRoot $ArchiveRoot
}

$baselineKdAbs = Normalize-Path -P $KeepBaselineKD
$baselineDkdAbs = Normalize-Path -P $KeepBaselineDKD

$mdAbs = @()
foreach ($p in $KeepFromMd) {
    $q = $p
    if (-not [System.IO.Path]::IsPathRooted($q)) { $q = Join-Path $RepoRoot $p }
    $mdAbs += $q
}

$keepNames = Extract-KeepNamesFromMarkdown -MdPaths $mdAbs
if ($baselineKdAbs) { [void]$keepNames.Add((Split-Path -Leaf $baselineKdAbs)) }
if ($baselineDkdAbs) { [void]$keepNames.Add((Split-Path -Leaf $baselineDkdAbs)) }

Write-Host "RepoRoot      : $RepoRoot"
Write-Host "StudentsRoot  : $studentsAbs"
Write-Host "ArchiveRoot   : $archiveAbs"
Write-Host "Zip           : $Zip"
Write-Host "RemoveAfterZip: $RemoveAfterZip"
Write-Host "DryRun        : $DryRun"
Write-Host "KeepNames     : $($keepNames.Count) referenced/baseline runs"

$targets = @()
foreach ($mode in @("KD","DKD")) {
    $modeRoot = Join-Path $studentsAbs $mode
    if (-not (Test-Path -LiteralPath $modeRoot)) { continue }

    $runs = Get-ChildItem -LiteralPath $modeRoot -Directory |
        Where-Object { $_.Name -like "*_" + $mode + "_*" }

    foreach ($r in $runs) {
        if ($keepNames.Contains($r.Name)) { continue }
        $targets += [pscustomobject]@{
            Mode = $mode
            Name = $r.Name
            FullName = $r.FullName
            DateKey = (Get-RunDateKey -RunName $r.Name)
        }
    }
}

$targets = $targets | Sort-Object DateKey, Mode, Name
Write-Host "Archive candidates: $($targets.Count)"

if ($targets.Count -eq 0) {
    Write-Host "Nothing to archive."
    exit 0
}

# Group by date for reporting
$byDate = $targets | Group-Object DateKey
foreach ($g in $byDate) {
    Write-Host ("  " + $g.Name + ": " + $g.Count)
}

if ($DryRun) {
    Write-Host "\nDry-run: would archive these folders:"
    $targets | Select-Object Mode, DateKey, Name, FullName | Format-Table -AutoSize | Out-String | Write-Host
    exit 0
}

Ensure-Dir -Path $archiveAbs

foreach ($t in $targets) {
    $dateDir = Join-Path $archiveAbs $t.DateKey
    $modeDir = Join-Path $dateDir $t.Mode
    Ensure-Dir -Path $modeDir

    $destDir = Join-Path $modeDir $t.Name

    Write-Host "\n[archive] $($t.FullName) -> $destDir"
    Move-Item -LiteralPath $t.FullName -Destination $destDir

    if ($Zip) {
        $zipDir = Join-Path $modeDir "_zips"
        Ensure-Dir -Path $zipDir
        $zipPath = Join-Path $zipDir ("$($t.Name).zip")

        Write-Host "[zip] $destDir -> $zipPath"
        if (Test-Path -LiteralPath $zipPath) {
            Remove-Item -LiteralPath $zipPath -Force
        }
        Compress-Archive -Path $destDir -DestinationPath $zipPath -Force

        if ($RemoveAfterZip) {
            Write-Host "[remove] $destDir"
            Remove-Item -LiteralPath $destDir -Recurse -Force
        }
    }
}

Write-Host "\nDone."