param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

$ErrorActionPreference = "Stop"

$frontendDirectory = Join-Path $ProjectRoot "frontend"
if (!(Test-Path $frontendDirectory)) {
    throw "Frontend directory not found: $frontendDirectory"
}

Push-Location $frontendDirectory
try {
    if (!(Test-Path "node_modules")) {
        Write-Host "[frontend] Installing npm dependencies..."
        npm install
    }

    Write-Host "[frontend] Building Tauri installer..."
    npm run tauri:build

    $msiPath = Join-Path $frontendDirectory "src-tauri\target\release\bundle\msi\Anantum Siri Widget_0.1.0_x64_en-US.msi"
    $nsisPath = Join-Path $frontendDirectory "src-tauri\target\release\bundle\nsis\Anantum Siri Widget_0.1.0_x64-setup.exe"

    if (!(Test-Path $msiPath)) {
        throw "MSI artifact not found after build: $msiPath"
    }
    if (!(Test-Path $nsisPath)) {
        throw "NSIS artifact not found after build: $nsisPath"
    }

    Write-Host "[frontend] MSI : $msiPath"
    Write-Host "[frontend] NSIS: $nsisPath"
} finally {
    Pop-Location
}
