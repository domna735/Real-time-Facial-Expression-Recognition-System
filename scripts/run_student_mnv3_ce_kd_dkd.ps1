param(
    [string]$RepoRoot = "",

    # Model/data
    [string]$Model = "mobilenetv3_large_100",
    [string]$Manifest = "Training_data_cleaned/classification_manifest_hq_train.csv",
    [string]$DataRoot = "Training_data_cleaned",
    [int]$ImageSize = 224,
    [switch]$UseClahe,

    # Softlabels (for KD/DKD)
    [string]$SoftlabelsDir = "outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856",

    # Training
    [int]$BatchSize = 256,
    [int]$NumWorkers = 2,
    [switch]$UseAmp,

    # CE/KD/DKD schedule
    [int]$CeEpochs = 10,
    [int]$KdEpochs = 20,
    [int]$DkdEpochs = 10,

    # KD/DKD hyperparams
    [double]$Temperature = 2.0,
    [double]$Alpha = 0.5,
    [double]$Beta = 4.0,

    # Smoke test controls
    [switch]$SmokeOnly,
    [switch]$SkipSmoke,
    [int]$SmokeEpochs = 1,
    [int]$SmokeBatchSize = 64,
    [int]$SmokeMaxValBatches = 2,

    # Repro
    [int]$Seed = 1337
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    param([string]$Provided)
    if ($Provided -and (Test-Path -LiteralPath $Provided)) {
        return (Resolve-Path -LiteralPath $Provided).Path
    }
    $here = Split-Path -Parent $PSCommandPath
    $root = Resolve-Path -LiteralPath (Join-Path $here "..")
    return $root.Path
}

function Resolve-PythonExe {
    param([string]$Root)
    $py = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $py) {
        return $py
    }
    $pyAlt = Join-Path $Root ".venv\Scripts\python"
    if (Test-Path -LiteralPath $pyAlt) {
        return $pyAlt
    }
    return "python"
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Run-Student {
    param(
        [string]$PythonExe,
        [string]$Root,
        [string]$Label,
        [string[]]$ArgList,
        [string]$LogPath
    )

    Write-Host "\n==== $Label ===="
    Write-Host ("python " + ($ArgList -join " "))

    Ensure-Dir (Split-Path -Parent $LogPath)

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonExe
    $psi.WorkingDirectory = $Root
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.Arguments = ($ArgList -join " ")

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi

    $null = $p.Start()

    $outLines = New-Object System.Collections.Generic.List[string]

    while (-not $p.HasExited) {
        Start-Sleep -Milliseconds 200
        while (-not $p.StandardOutput.EndOfStream) {
            $line = $p.StandardOutput.ReadLine()
            if ($null -ne $line) {
                Write-Host $line
                $outLines.Add($line) | Out-Null
            }
        }
        while (-not $p.StandardError.EndOfStream) {
            $line = $p.StandardError.ReadLine()
            if ($null -ne $line) {
                Write-Host $line
                $outLines.Add($line) | Out-Null
            }
        }
    }

    # Drain remaining
    while (-not $p.StandardOutput.EndOfStream) {
        $line = $p.StandardOutput.ReadLine()
        if ($null -ne $line) {
            Write-Host $line
            $outLines.Add($line) | Out-Null
        }
    }
    while (-not $p.StandardError.EndOfStream) {
        $line = $p.StandardError.ReadLine()
        if ($null -ne $line) {
            Write-Host $line
            $outLines.Add($line) | Out-Null
        }
    }

    $code = $p.ExitCode
    $outLines | Set-Content -LiteralPath $LogPath -Encoding UTF8

    if ($code -ne 0) {
        throw "Run failed ($Label) with exit code $code. See log: $LogPath"
    }

    return $code
}

$RepoRoot = Resolve-RepoRoot -Provided $RepoRoot
$PythonExe = Resolve-PythonExe -Root $RepoRoot

$stamp = Get-Date -Format yyyyMMdd_HHmmss
$logRoot = Join-Path $RepoRoot ("outputs/students/_logs_" + $stamp)
Ensure-Dir $logRoot

$softlabelsDirAbs = Join-Path $RepoRoot $SoftlabelsDir
$softlabelsPath = Join-Path $softlabelsDirAbs "softlabels.npz"
$softlabelsIndexPath = Join-Path $softlabelsDirAbs "softlabels_index.jsonl"

if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot $Manifest))) {
    throw "Manifest not found: $(Join-Path $RepoRoot $Manifest)"
}

if (-not $SmokeOnly) {
    # For full run, require softlabels dir because we will run KD/DKD.
    if (-not (Test-Path -LiteralPath $softlabelsPath)) {
        throw "softlabels.npz not found: $softlabelsPath"
    }
    if (-not (Test-Path -LiteralPath $softlabelsIndexPath)) {
        throw "softlabels_index.jsonl not found: $softlabelsIndexPath"
    }
}

# Fixed output dirs so DKD can resume automatically
$ceOut = Join-Path $RepoRoot ("outputs/students/${Model}_img${ImageSize}_seed${Seed}_CE_" + $stamp)
$kdOut = Join-Path $RepoRoot ("outputs/students/${Model}_img${ImageSize}_seed${Seed}_KD_" + $stamp)
$dkdOut = Join-Path $RepoRoot ("outputs/students/${Model}_img${ImageSize}_seed${Seed}_DKD_" + $stamp)

function Build-CommonArgs {
    param([int]$Epochs,[int]$Bs,[int]$Workers,[string]$Mode,[string]$OutDir,[int]$MaxValBatches)

    $args = @(
        "scripts/train_student.py",
        "--mode", $Mode,
        "--model", $Model,
        "--manifest", $Manifest,
        "--data-root", $DataRoot,
        "--image-size", $ImageSize,
        "--batch-size", $Bs,
        "--num-workers", $Workers,
        "--epochs", $Epochs,
        "--eval-every", 1,
        "--seed", $Seed,
        "--output-dir", $OutDir
    )

    if ($UseClahe) {
        $args += "--use-clahe"
    }
    if ($UseAmp) {
        $args += "--use-amp"
    }
    if ($MaxValBatches -gt 0) {
        $args += @("--max-val-batches", $MaxValBatches)
    }

    return $args
}

function Build-KDArgs {
    param([string]$Mode,[int]$Epochs,[int]$Bs,[int]$Workers,[string]$OutDir,[int]$MaxValBatches,[string]$ResumeCkpt)

    $args = Build-CommonArgs -Epochs $Epochs -Bs $Bs -Workers $Workers -Mode $Mode -OutDir $OutDir -MaxValBatches $MaxValBatches

    $args += @(
        "--softlabels", $softlabelsPath,
        "--softlabels-index", $softlabelsIndexPath,
        "--temperature", $Temperature,
        "--alpha", $Alpha
    )

    if ($Mode -eq "dkd") {
        $args += @("--beta", $Beta)
        if ($ResumeCkpt) {
            $args += @("--resume", $ResumeCkpt)
        }
    }

    return $args
}

function Get-CkptEpoch {
    param(
        [string]$PythonExe,
        [string]$Root,
        [string]$CkptPath
    )
    if (-not (Test-Path -LiteralPath $CkptPath)) {
        return -1
    }
    Push-Location $Root
    try {
        $code = "import sys, torch; p=sys.argv[1]; ck=torch.load(p, map_location='cpu', weights_only=False); print(int(ck.get('epoch', -1)))"
        $out = & $PythonExe -c $code $CkptPath 2>$null
        if ($null -eq $out) {
            return -1
        }
        return [int]($out.ToString().Trim())
    } catch {
        return -1
    } finally {
        Pop-Location
    }
}

# 1) Smoke test (fast sanity)
if (-not $SkipSmoke) {
    $smokeStamp = Get-Date -Format yyyyMMdd_HHmmss
    $smokeCeOut = Join-Path $RepoRoot ("outputs/students/_smoke_${Model}_CE_" + $smokeStamp)
    $smokeKdOut = Join-Path $RepoRoot ("outputs/students/_smoke_${Model}_KD_" + $smokeStamp)
    $smokeDkdOut = Join-Path $RepoRoot ("outputs/students/_smoke_${Model}_DKD_" + $smokeStamp)

    $ceArgs = Build-CommonArgs -Epochs $SmokeEpochs -Bs $SmokeBatchSize -Workers 2 -Mode "ce" -OutDir $smokeCeOut -MaxValBatches $SmokeMaxValBatches
    Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "SMOKE CE" -ArgList $ceArgs -LogPath (Join-Path $logRoot "smoke_ce.txt")

    if (-not (Test-Path -LiteralPath $softlabelsPath)) {
        Write-Host "Skipping SMOKE KD/DKD because softlabels not found: $softlabelsPath"
    } else {
        $kdArgs = Build-KDArgs -Mode "kd" -Epochs $SmokeEpochs -Bs $SmokeBatchSize -Workers 2 -OutDir $smokeKdOut -MaxValBatches $SmokeMaxValBatches -ResumeCkpt ""
        Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "SMOKE KD" -ArgList $kdArgs -LogPath (Join-Path $logRoot "smoke_kd.txt")

        $bestCkpt = Join-Path $smokeKdOut "best.pt"
        if (-not (Test-Path -LiteralPath $bestCkpt)) {
            $bestCkpt = Join-Path $smokeKdOut "checkpoint_last.pt"
        }

        $dkdArgs = Build-KDArgs -Mode "dkd" -Epochs $SmokeEpochs -Bs $SmokeBatchSize -Workers 2 -OutDir $smokeDkdOut -MaxValBatches $SmokeMaxValBatches -ResumeCkpt $bestCkpt
        Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "SMOKE DKD" -ArgList $dkdArgs -LogPath (Join-Path $logRoot "smoke_dkd.txt")
    }

    if ($SmokeOnly) {
        Write-Host "\nSmoke-only requested; stopping here. Logs: $logRoot"
        exit 0
    }
}

# 2) Full runs
$ceArgsFull = Build-CommonArgs -Epochs $CeEpochs -Bs $BatchSize -Workers $NumWorkers -Mode "ce" -OutDir $ceOut -MaxValBatches 0
Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "FULL CE" -ArgList $ceArgsFull -LogPath (Join-Path $logRoot "full_ce.txt")

$kdArgsFull = Build-KDArgs -Mode "kd" -Epochs $KdEpochs -Bs $BatchSize -Workers $NumWorkers -OutDir $kdOut -MaxValBatches 0 -ResumeCkpt ""
Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "FULL KD" -ArgList $kdArgsFull -LogPath (Join-Path $logRoot "full_kd.txt")

$kdBest = Join-Path $kdOut "best.pt"
if (-not (Test-Path -LiteralPath $kdBest)) {
    $kdBest = Join-Path $kdOut "checkpoint_last.pt"
}

$dkdArgsFull = Build-KDArgs -Mode "dkd" -Epochs $DkdEpochs -Bs $BatchSize -Workers $NumWorkers -OutDir $dkdOut -MaxValBatches 0 -ResumeCkpt $kdBest
$dkdTotalEpochs = $DkdEpochs
if ($kdBest -and (Test-Path -LiteralPath $kdBest)) {
    $kdEpoch = Get-CkptEpoch -PythonExe $PythonExe -Root $RepoRoot -CkptPath $kdBest
    if ($kdEpoch -ge 0) {
        # Resume starts at (kdEpoch+1). We want DkdEpochs additional epochs.
        $dkdTotalEpochs = $kdEpoch + 1 + $DkdEpochs
    }
}
$dkdArgsFull = Build-KDArgs -Mode "dkd" -Epochs $dkdTotalEpochs -Bs $BatchSize -Workers $NumWorkers -OutDir $dkdOut -MaxValBatches 0 -ResumeCkpt $kdBest
Run-Student -PythonExe $PythonExe -Root $RepoRoot -Label "FULL DKD" -ArgList $dkdArgsFull -LogPath (Join-Path $logRoot "full_dkd.txt")

Write-Host "\nAll done. Outputs:" 
Write-Host "  CE : $ceOut"
Write-Host "  KD : $kdOut"
Write-Host "  DKD: $dkdOut"
Write-Host "Logs: $logRoot"
