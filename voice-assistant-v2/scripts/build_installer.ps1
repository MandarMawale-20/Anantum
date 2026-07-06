param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Running preflight checks"
& "$PSScriptRoot\preflight.ps1" -ProjectRoot $ProjectRoot

Write-Host "[2/3] Building Python backend executable"
& "$PSScriptRoot\build_backend.ps1" -ProjectRoot $ProjectRoot

Write-Host "[3/3] Building Tauri installer"
& "$PSScriptRoot\build_frontend.ps1" -ProjectRoot $ProjectRoot

$backendExe = Join-Path $ProjectRoot "frontend\dist\backend\assistant-backend\assistant-backend.exe"
if (!(Test-Path $backendExe)) {
    throw "Backend executable missing after build: $backendExe"
}

Write-Host "Build complete."
Write-Host "[summary] Backend: $backendExe"
Write-Host "[summary] Installer artifacts: frontend/src-tauri/target/release/bundle"
