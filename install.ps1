[CmdletBinding()]
param(
    [string]$ComfyPath = "",
    [switch]$Yes,
    [switch]$DryRun,
    [string]$ReleaseTag = "v0.2.0"
)

$ErrorActionPreference = "Stop"

# Ensure TLS 1.2 for GitHub downloads on Windows PowerShell 5.1
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.ServicePointManager]::SecurityProtocol
} catch {
}

$RepoSlug = "wildersa/ComfyUI-Intel-Arc-XPU"
$ReleaseBaseUrl = "https://github.com/$RepoSlug/releases/download/$ReleaseTag"

$WheelSpecs = @(
    @{
        Dist = "omni-xpu-kernel"
        Version = "0.2.0b3+torch213.bmg"
        File = "omni_xpu_kernel-0.2.0b3+torch213.bmg-cp313-cp313-win_amd64.whl"
        Sha256 = "a5f3d152bce545c517a2e8f1db958c3c14d67238e7467cfa89bbbb859cb9fd25"
    },
    @{
        Dist = "comfy-kitchen-xpu-runtime"
        Version = "0.2.41"
        File = "comfy_kitchen_xpu_runtime-0.2.41-py3-none-any.whl"
        Sha256 = "638756e43526542b625d5a81717bc0721f503e53240416479b42922cde52c605"
    },
    @{
        Dist = "comfy-aimdo-xpu-runtime"
        Version = "0.6.0"
        File = "comfy_aimdo_xpu_runtime-0.6.0-cp39-abi3-win_amd64.whl"
        Sha256 = "6f962d5bdc1f6be61e9ee1d4654b846ff3edea31e4c4fc4640ded6a405756981"
    }
)

$OfficialSpecs = @(
    @{ Dist = "comfy-aimdo"; Version = "0.6.0" },
    @{ Dist = "comfy-kitchen"; Version = "0.2.41" }
)

function Write-Info {
    param([string]$Message)
    Write-Host "[ComfyUI-Intel-Arc-XPU] $Message" -ForegroundColor Cyan
}

function Write-WarnMsg {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$DefaultYes = $true
    )
    if ($Yes) {
        return $true
    }
    $suffix = if ($DefaultYes) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $ans = Read-Host "$Prompt $suffix"
        if ([string]::IsNullOrWhiteSpace($ans)) {
            return $DefaultYes
        }
        $norm = $ans.Trim().ToLowerInvariant()
        if ($norm -in @("y", "yes")) { return $true }
        if ($norm -in @("n", "no")) { return $false }
        Write-Host "Please answer Y or N."
    }
}

function Resolve-ComfyLayout {
    param([string]$CandidatePath)
    if ([string]::IsNullOrWhiteSpace($CandidatePath)) {
        return $null
    }
    if (-not (Test-Path -LiteralPath $CandidatePath)) {
        return $null
    }
    $resolved = (Resolve-Path -LiteralPath $CandidatePath).Path

    $comfyRoot = $null
    $installRoot = $resolved

    if (Test-Path -LiteralPath (Join-Path $resolved "main.py")) {
        $comfyRoot = $resolved
        $parentDir = Split-Path -Path $resolved -Parent
        if ($parentDir) {
            $installRoot = $parentDir
        }
    } elseif (Test-Path -LiteralPath (Join-Path (Join-Path $resolved "ComfyUI") "main.py")) {
        $comfyRoot = Join-Path $resolved "ComfyUI"
        $installRoot = $resolved
    } else {
        return $null
    }

    $pyCandidates = @(
        (Join-Path (Join-Path (Join-Path $comfyRoot ".venv") "Scripts") "python.exe"),
        (Join-Path (Join-Path (Join-Path $installRoot ".venv") "Scripts") "python.exe"),
        (Join-Path (Join-Path $installRoot "python_embeded") "python.exe"),
        (Join-Path (Join-Path $comfyRoot "python_embeded") "python.exe")
    )

    $pythonExe = $null
    foreach ($py in $pyCandidates) {
        if (Test-Path -LiteralPath $py) {
            $pythonExe = $py
            break
        }
    }

    return [PSCustomObject]@{
        InstallRoot = $installRoot
        ComfyRoot   = $comfyRoot
        CustomNodes = (Join-Path $comfyRoot "custom_nodes")
        PythonExe   = $pythonExe
    }
}

function Find-ComfyInstallations {
    $candidates = New-Object System.Collections.Generic.List[string]

    $scriptParent = Split-Path -Path $PSScriptRoot -Parent
    if ($scriptParent -and ((Split-Path -Path $scriptParent -Leaf) -eq "custom_nodes")) {
        $possibleComfy = Split-Path -Path $scriptParent -Parent
        if ($possibleComfy) {
            $candidates.Add($possibleComfy)
        }
    }

    $desktopDir = Join-Path $env:APPDATA "Comfy Desktop"
    $instJson = Join-Path $desktopDir "installations.json"
    if (Test-Path -LiteralPath $instJson) {
        try {
            $items = Get-Content -LiteralPath $instJson -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($item in @($items)) {
                if ($item -and $item.installPath -and ($item.sourceId -ne "cloud")) {
                    $candidates.Add([string]$item.installPath)
                }
            }
        } catch {
            Write-WarnMsg "Could not parse ${instJson}: $($_.Exception.Message)"
        }
    }

    foreach ($cfgName in @("config.json", "settings.json")) {
        $cfgPath = Join-Path $desktopDir $cfgName
        if (Test-Path -LiteralPath $cfgPath) {
            try {
                $cfg = Get-Content -LiteralPath $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($cfg.installPath) {
                    $candidates.Add([string]$cfg.installPath)
                }
                if ($cfg.installDir) {
                    $candidates.Add((Join-Path ([string]$cfg.installDir) "ComfyUI"))
                }
            } catch {
            }
        }
    }

    $commonRoots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\ComfyUI"),
        (Join-Path $env:USERPROFILE "ComfyUI"),
        "C:\ComfyUI",
        "D:\ComfyUI",
        "E:\ComfyUI",
        "C:\ComfyUI_windows_portable",
        "D:\ComfyUI_windows_portable",
        "E:\ComfyUI_windows_portable"
    )
    foreach ($root in $commonRoots) {
        $candidates.Add($root)
    }

    $seen = @{}
    $found = New-Object System.Collections.Generic.List[object]
    foreach ($c in $candidates) {
        $layout = Resolve-ComfyLayout -CandidatePath $c
        if ($layout -and (-not $seen.ContainsKey($layout.ComfyRoot.ToLowerInvariant()))) {
            $seen[$layout.ComfyRoot.ToLowerInvariant()] = $true
            $found.Add($layout)
        }
    }
    return $found
}

function Select-ComfyInstallation {
    if (-not [string]::IsNullOrWhiteSpace($ComfyPath)) {
        $explicit = Resolve-ComfyLayout -CandidatePath $ComfyPath
        if (-not $explicit) {
            throw "Could not find ComfyUI (main.py) at -ComfyPath '$ComfyPath'."
        }
        return $explicit
    }

    $detected = @(Find-ComfyInstallations)
    if ($detected.Count -eq 1) {
        return $detected[0]
    }
    if ($detected.Count -gt 1) {
        if ($Yes) {
            return $detected[0]
        }
        Write-Host ""
        Write-Host "Multiple ComfyUI installations detected:"
        for ($i = 0; $i -lt $detected.Count; $i++) {
            $item = $detected[$i]
            Write-Host ("  [{0}] {1} (Python: {2})" -f ($i + 1), $item.ComfyRoot, $item.PythonExe)
        }
        while ($true) {
            $choice = Read-Host "Select installation number [1]"
            if ([string]::IsNullOrWhiteSpace($choice)) {
                return $detected[0]
            }
            $idx = 0
            if ([int]::TryParse($choice, [ref]$idx) -and $idx -ge 1 -and $idx -le $detected.Count) {
                return $detected[$idx - 1]
            }
            Write-Host "Invalid selection."
        }
    }

    if ($Yes -or $DryRun) {
        throw "No ComfyUI installation auto-detected. Pass -ComfyPath <path-to-ComfyUI>."
    }

    while ($true) {
        $userPath = Read-Host "Enter path to your ComfyUI or ComfyUI_windows_portable directory"
        $layout = Resolve-ComfyLayout -CandidatePath $userPath
        if ($layout) {
            return $layout
        }
        Write-WarnMsg "Could not find main.py under '$userPath'."
    }
}

function Get-RunningComfyProcesses {
    param([object]$Layout)
    $hits = New-Object System.Collections.Generic.List[string]
    $normComfy = $Layout.ComfyRoot.ToLowerInvariant()
    $normInstall = $Layout.InstallRoot.ToLowerInvariant()

    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
        foreach ($p in $procs) {
            $name = ([string]$p.Name).ToLowerInvariant()
            $exe = ([string]$p.ExecutablePath).ToLowerInvariant()
            $cmd = ([string]$p.CommandLine).ToLowerInvariant()

            if ($name -like "*comfy*desktop*") {
                $hits.Add("$($p.Name) (PID $($p.ProcessId))")
                continue
            }
            if ($name -like "python*") {
                if (($exe -and ($exe.StartsWith($normComfy) -or $exe.StartsWith($normInstall))) -or
                    ($cmd -and ($cmd.Contains($normComfy) -or $cmd.Contains($normInstall)))) {
                    $hits.Add("$($p.Name) (PID $($p.ProcessId))")
                }
            }
        }
    } catch {
        $desktopProcs = Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -like "*Comfy*" }
        foreach ($dp in $desktopProcs) {
            $hits.Add("$($dp.ProcessName) (PID $($dp.Id))")
        }
    }
    return $hits
}

function Wait-ForComfyShutdown {
    param([object]$Layout)
    while ($true) {
        $running = @(Get-RunningComfyProcesses -Layout $Layout)
        if ($running.Count -eq 0) {
            return
        }
        Write-WarnMsg "Comfy Desktop / ComfyUI appears to be running:"
        foreach ($r in $running) {
            Write-Host "  - $r" -ForegroundColor Yellow
        }
        Write-Host "Please close Comfy Desktop / ComfyUI before installing wheels so Windows does not lock DLL/PYD files."
        if ($DryRun) {
            Write-WarnMsg "DryRun mode: continuing read-only inspection despite running process."
            return
        }
        if ($Yes) {
            throw "Comfy Desktop / ComfyUI is currently running. Close it and re-run install.ps1."
        }
        $ans = Read-Host "Close ComfyUI and press [R]etry, or [A]bort [R/A]"
        if (-not [string]::IsNullOrWhiteSpace($ans) -and $ans.Trim().ToLowerInvariant().StartsWith("a")) {
            throw "Installation aborted by user while ComfyUI was running."
        }
    }
}

function Load-Sha256Sums {
    param([string]$SumsFile)
    $map = @{}
    if (Test-Path -LiteralPath $SumsFile) {
        foreach ($rawLine in (Get-Content -LiteralPath $SumsFile -Encoding UTF8)) {
            $line = $rawLine.Trim()
            if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) { continue }
            $parts = $line -split "\s+"
            if ($parts.Length -ge 2) {
                $hash = $parts[0].ToLowerInvariant()
                $file = $parts[$parts.Length - 1].TrimStart("*")
                if ($hash.Length -eq 64) {
                    $map[$file] = $hash
                }
            }
        }
    }
    return $map
}

function Verify-FileSha256 {
    param(
        [string]$Path,
        [string]$ExpectedSha256
    )
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $(Split-Path -Path $Path -Leaf): expected $ExpectedSha256, got $actual"
    }
}

function Find-UvExecutable {
    param([string]$PythonExe)
    $pyDir = Split-Path -Path $PythonExe -Parent
    $candidates = @(
        (Join-Path $pyDir "uv.exe"),
        (Join-Path (Split-Path -Path (Split-Path -Path $pyDir -Parent) -Parent) "standalone-env\uv.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            return $c
        }
    }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Invoke-PackageInstall {
    param(
        [string]$PythonExe,
        [string]$UvExe,
        [string[]]$Targets,
        [bool]$ForceReinstall
    )
    if ($Targets.Count -eq 0) {
        return
    }
    if ($UvExe) {
        $uvArgs = @("pip", "install", "--python", $PythonExe, "--no-deps")
        if ($ForceReinstall) { $uvArgs += "--reinstall" }
        $uvArgs += $Targets
        Write-Info ("Running: {0} {1}" -f $UvExe, ($uvArgs -join " "))
        & $UvExe @uvArgs
        if ($LASTEXITCODE -ne 0) {
            throw "uv pip install failed with exit code $LASTEXITCODE"
        }
    } else {
        $pipArgs = @("-m", "pip", "install", "--no-deps")
        if ($ForceReinstall) { $pipArgs += "--force-reinstall" }
        $pipArgs += $Targets
        Write-Info ("Running: {0} {1}" -f $PythonExe, ($pipArgs -join " "))
        & $PythonExe @pipArgs
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed with exit code $LASTEXITCODE"
        }
    }
}

# --- Main ---

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " ComfyUI-Intel-Arc-XPU ($ReleaseTag) Windows Installer" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not [System.Environment]::Is64BitOperatingSystem) {
    throw "Windows x64 is required."
}

$layout = Select-ComfyInstallation
if (-not $layout.PythonExe -or -not (Test-Path -LiteralPath $layout.PythonExe)) {
    if ($Yes -or $DryRun) {
        throw "Could not locate python.exe for ComfyUI at '$($layout.ComfyRoot)'."
    }
    $manualPy = Read-Host "Could not auto-detect python.exe. Enter full path to ComfyUI python.exe"
    if (-not (Test-Path -LiteralPath $manualPy)) {
        throw "Python executable not found at '$manualPy'."
    }
    $layout.PythonExe = (Resolve-Path -LiteralPath $manualPy).Path
}

Wait-ForComfyShutdown -Layout $layout

Write-Info "Probing Python and PyTorch environment at $($layout.PythonExe)..."
$probeScript = @"
import importlib.metadata as m, json, platform, struct, sys
def ver(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
print(json.dumps({
    'py_major': sys.version_info[0],
    'py_minor': sys.version_info[1],
    'py_full': sys.version.split()[0],
    'bits': struct.calcsize('P') * 8,
    'machine': platform.machine(),
    'torch': ver('torch'),
    'installed': {
        'comfy-aimdo': ver('comfy-aimdo'),
        'comfy-kitchen': ver('comfy-kitchen'),
        'omni-xpu-kernel': ver('omni-xpu-kernel'),
        'comfy-kitchen-xpu-runtime': ver('comfy-kitchen-xpu-runtime'),
        'comfy-aimdo-xpu-runtime': ver('comfy-aimdo-xpu-runtime'),
    }
}))
"@
$probeRaw = & $layout.PythonExe -c $probeScript
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect Python environment at '$($layout.PythonExe)'."
}
$probe = $probeRaw | ConvertFrom-Json

if ($probe.bits -ne 64) {
    throw "64-bit Python is required (detected $($probe.bits)-bit)."
}
if (($probe.py_major -ne 3) -or ($probe.py_minor -ne 13)) {
    throw "Python 3.13 (cp313) is required for prebuilt wheels; detected Python $($probe.py_full)."
}
if (-not $probe.torch -or (-not ([string]$probe.torch).StartsWith("2.13.0+xpu"))) {
    throw @"
PyTorch 2.13.0+xpu is required (detected torch=$($probe.torch)).
Upgrade ComfyUI's Python environment first:
  "$($layout.PythonExe)" -m pip install --upgrade torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
"@
}
Write-Ok "Environment validated: Windows x64, Python $($probe.py_full), torch==$($probe.torch)"

# Check SHA256SUMS.txt
$sumsMap = Load-Sha256Sums -SumsFile (Join-Path $PSScriptRoot "SHA256SUMS.txt")
foreach ($w in $WheelSpecs) {
    if ($sumsMap.ContainsKey($w.File) -and ($sumsMap[$w.File] -ne $w.Sha256)) {
        throw "SHA256SUMS.txt mismatch for $($w.File)."
    }
}

# Plan official and wheel installs
$officialTodo = @()
foreach ($spec in $OfficialSpecs) {
    $cur = $probe.installed.($spec.Dist)
    if ($cur -ne $spec.Version) {
        $officialTodo += [PSCustomObject]@{ Dist = $spec.Dist; Current = $cur; Target = $spec.Version }
    }
}

$wheelsTodo = @()
$localWheelsDir = Join-Path $PSScriptRoot "wheels"
foreach ($w in $WheelSpecs) {
    $cur = $probe.installed.($w.Dist)
    if ($cur -ne $w.Version) {
        $localWhl = Join-Path $localWheelsDir $w.File
        $sourceLabel = if (Test-Path -LiteralPath $localWhl) { "local wheels\$($w.File)" } else { "$ReleaseBaseUrl/$($w.File)" }
        $wheelsTodo += [PSCustomObject]@{
            Dist = $w.Dist
            Current = $cur
            Target = $w.Version
            File = $w.File
            Sha256 = $w.Sha256
            LocalPath = $localWhl
            Source = $sourceLabel
        }
    }
}

$targetNodeDir = Join-Path $layout.CustomNodes "ComfyUI-Intel-Arc-XPU"
$sourceResolved = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd("\")
$targetAlreadyInPlace = ($sourceResolved.ToLowerInvariant() -eq $targetNodeDir.ToLowerInvariant())
$conflictBackend = Join-Path $layout.CustomNodes "comfyui-xpu-backend"
$hasConflictBackend = Test-Path -LiteralPath $conflictBackend

Write-Host ""
Write-Host "Planned actions:" -ForegroundColor White
Write-Host "  ComfyUI Root   : $($layout.ComfyRoot)"
Write-Host "  Python         : $($layout.PythonExe)"
Write-Host "  Custom Node    : $targetNodeDir"
if ($targetAlreadyInPlace) {
    Write-Host "  Extension Copy : Already inside target custom_nodes directory (preserve in place)"
} else {
    Write-Host "  Extension Copy : Copy files from $sourceResolved -> $targetNodeDir"
}
if ($officialTodo.Count -eq 0 -and $wheelsTodo.Count -eq 0) {
    Write-Host "  Python Packages: All 5 target packages already at exact versions (skipping reinstall)"
} else {
    foreach ($o in $officialTodo) {
        Write-Host ("  - Install {0}=={1} (current: {2})" -f $o.Dist, $o.Target, $(if ($o.Current) { $o.Current } else { "not installed" }))
    }
    foreach ($w in $wheelsTodo) {
        Write-Host ("  - Install {0}=={1} (current: {2}) from {3}" -f $w.Dist, $w.Target, $(if ($w.Current) { $w.Current } else { "not installed" }), $w.Source)
    }
}
if ($hasConflictBackend) {
    Write-WarnMsg "Conflicting custom node found: $conflictBackend"
}

if ($DryRun) {
    Write-Host ""
    Write-Ok "DryRun complete. No changes were made."
    exit 0
}

if (-not (Read-YesNo -Prompt "Proceed with installation?" -DefaultYes $true)) {
    Write-Info "Installation cancelled."
    exit 0
}

# Handle conflicting custom_nodes\comfyui-xpu-backend if present
if ($hasConflictBackend -and (-not $Yes)) {
    Write-Host ""
    Write-WarnMsg "Found legacy 'comfyui-xpu-backend' at $conflictBackend. It conflicts with OmniXPU."
    $cAns = Read-Host "Action for comfyui-xpu-backend: [D]isable (rename .disabled), [R]emove, or [S]kip? [D/r/s]"
    $cNorm = if ($cAns) { $cAns.Trim().ToLowerInvariant() } else { "d" }
    if ($cNorm -eq "d") {
        $disabledPath = "$conflictBackend.disabled"
        if (Test-Path -LiteralPath $disabledPath) {
            $disabledPath = "$conflictBackend.disabled.$(Get-Date -Format 'yyyyMMddHHmmss')"
        }
        Move-Item -LiteralPath $conflictBackend -Destination $disabledPath
        Write-Ok "Renamed conflicting node to $disabledPath"
    } elseif ($cNorm -eq "r") {
        Remove-Item -LiteralPath $conflictBackend -Recurse -Force
        Write-Ok "Removed $conflictBackend"
    } else {
        Write-WarnMsg "Skipped removing $conflictBackend; remember to disable it manually."
    }
}

$uvExe = Find-UvExecutable -PythonExe $layout.PythonExe

if ($officialTodo.Count -gt 0) {
    $officialArgs = @()
    $needForceOfficial = $false
    foreach ($o in $officialTodo) {
        $officialArgs += "$($o.Dist)==$($o.Target)"
        if ($o.Current) { $needForceOfficial = $true }
    }
    Invoke-PackageInstall -PythonExe $layout.PythonExe -UvExe $uvExe -Targets $officialArgs -ForceReinstall $needForceOfficial
}

if ($wheelsTodo.Count -gt 0) {
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("omnixpu_whl_" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
    try {
        $wheelPaths = @()
        $needForceWheels = $false
        foreach ($w in $wheelsTodo) {
            if ($w.Current) { $needForceWheels = $true }
            if (Test-Path -LiteralPath $w.LocalPath) {
                Write-Info "Verifying local wheel $($w.File)..."
                Verify-FileSha256 -Path $w.LocalPath -ExpectedSha256 $w.Sha256
                $wheelPaths += $w.LocalPath
            } else {
                $url = "$ReleaseBaseUrl/$($w.File)"
                $dest = Join-Path $tempDir $w.File
                Write-Info "Downloading $url ..."
                Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
                Verify-FileSha256 -Path $dest -ExpectedSha256 $w.Sha256
                Write-Ok "Verified SHA-256 for $($w.File)"
                $wheelPaths += $dest
            }
        }
        Invoke-PackageInstall -PythonExe $layout.PythonExe -UvExe $uvExe -Targets $wheelPaths -ForceReinstall $needForceWheels
    } finally {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not $targetAlreadyInPlace) {
    if (-not (Test-Path -LiteralPath $layout.CustomNodes)) {
        New-Item -ItemType Directory -Path $layout.CustomNodes -Force | Out-Null
    }
    if (-not (Test-Path -LiteralPath $targetNodeDir)) {
        New-Item -ItemType Directory -Path $targetNodeDir -Force | Out-Null
    }
    Write-Info "Copying extension files to $targetNodeDir ..."
    Get-ChildItem -LiteralPath $sourceResolved -Recurse -File | Where-Object {
        $_.FullName -notmatch '\\\.git(\\|$)|\\\.venv(\\|$)|\\__pycache__(\\|$)|\\wheels(\\|$)' -and
        $_.Extension -ne ".whl"
    } | ForEach-Object {
        $rel = $_.FullName.Substring($sourceResolved.Length + 1)
        $destFile = Join-Path $targetNodeDir $rel
        $destParent = Split-Path -Path $destFile -Parent
        if (-not (Test-Path -LiteralPath $destParent)) {
            New-Item -ItemType Directory -Path $destParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $destFile -Force
    }
    Write-Ok "Extension files updated in $targetNodeDir"
}

Write-Info "Running post-install verification..."
$verifyScript = @"
import importlib.metadata as m
import torch
import omni_xpu_kernel
from omni_xpu_kernel import cute
import comfy_kitchen
import comfy_aimdo

print('torch:', torch.__version__, 'xpu_available:', torch.xpu.is_available())
print('omni_xpu_kernel:', omni_xpu_kernel.__version__, 'cute_available:', cute.is_available())
print('comfy-kitchen:', m.version('comfy-kitchen'), '| runtime:', m.version('comfy-kitchen-xpu-runtime'))
print('comfy-aimdo:', m.version('comfy-aimdo'), '| runtime:', m.version('comfy-aimdo-xpu-runtime'))
"@
& $layout.PythonExe -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    throw "Post-install verification failed."
}
Write-Ok "Verification passed."

$currentAttn = [System.Environment]::GetEnvironmentVariable("OMNI_ATTN_BACKEND", "User")
if ($currentAttn -ne "cute") {
    Write-Host ""
    Write-Host "Windows defaults OmniXPU attention to PyTorch SDPA unless OMNI_ATTN_BACKEND=cute is set."
    if (Read-YesNo -Prompt "Set User environment variable OMNI_ATTN_BACKEND=cute now?" -DefaultYes $true) {
        [System.Environment]::SetEnvironmentVariable("OMNI_ATTN_BACKEND", "cute", "User")
        Write-Ok "Set User environment variable OMNI_ATTN_BACKEND=cute (restart Comfy Desktop / terminals to take effect)."
    }
} else {
    Write-Ok "User environment variable OMNI_ATTN_BACKEND is already set to 'cute'."
}

Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Ensure ComfyUI launches with: --enable-dynamic-vram --reserve-vram 4"
Write-Host "2. Restart Comfy Desktop / ComfyUI so new environment variables and prestartup hooks load."
Write-Host "3. Use the 'OmniXPU Status' node in a workflow (or check comfyui.log) to verify XPU runtime providers."
