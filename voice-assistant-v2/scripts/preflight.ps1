param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

$ErrorActionPreference = "Stop"

function Assert-Tool {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$VersionArgs = @("--version")
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "Missing required tool: $Name"
    }

    $versionOutput = ""
    try {
        $versionOutput = & $Name @VersionArgs 2>&1 | Select-Object -First 1
    } catch {
        $versionOutput = "version check failed"
    }

    if ([string]::IsNullOrWhiteSpace($versionOutput)) {
        $versionOutput = "version not reported"
    }

    Write-Host "[ok] $Name - $versionOutput"
}

Write-Host "Running packaging preflight checks..."
Write-Host "Project root: $ProjectRoot"

Assert-Tool -Name "python"
Assert-Tool -Name "rustc"
Assert-Tool -Name "cargo"
Assert-Tool -Name "node"
Assert-Tool -Name "npm"

$frontendDir = Join-Path $ProjectRoot "frontend"
if (!(Test-Path $frontendDir)) {
    throw "Frontend directory not found: $frontendDir"
}

Push-Location $frontendDir
try {
    try {
        $tauriVersion = npx --yes @tauri-apps/cli@latest --version 2>&1 | Select-Object -First 1
        Write-Host "[ok] tauri-cli - $tauriVersion"
    } catch {
        throw "Unable to execute Tauri CLI through npx. Install Node/npm dependencies first."
    }
} finally {
    Pop-Location
}

try {
    $pyiVersion = python -m PyInstaller --version 2>&1 | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($pyiVersion)) {
        throw "version not reported"
    }
    Write-Host "[ok] PyInstaller (python -m) - $pyiVersion"
} catch {
    throw "PyInstaller is not available through the active Python interpreter."
}

Write-Host "Preflight checks passed."
