param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\.."),
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

 $project = Resolve-Path $ProjectRoot
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $outputDirectory = Join-Path $project "frontend\dist\backend"
} else {
    $outputDirectory = $OutputDir
}

Write-Host "[backend] Project root: $project"
Write-Host "[backend] Output dir : $outputDirectory"

try {
    $pyiVersion = python -m PyInstaller --version 2>&1 | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($pyiVersion)) {
        throw "PyInstaller version check returned empty output"
    }
    Write-Host "[backend] PyInstaller: $pyiVersion"
} catch {
    throw "PyInstaller is not available in the active Python environment. Install with: python -m pip install pyinstaller"
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

Push-Location $project
try {
    Write-Host "[backend] Packaging backend executable with PyInstaller..."
        python -m PyInstaller `
      --noconfirm `
      --clean `
      --onedir `
      --name assistant-backend `
            --distpath "$outputDirectory" `
      --workpath "$project\.pyinstaller-build" `
      --specpath "$project\.pyinstaller-spec" `
      --collect-all faster_whisper `
      --collect-all sentence_transformers `
      --collect-all openwakeword `
      --paths "$project" `
      "$project\backend_launcher.py"

        $backendExecutable = Join-Path $outputDirectory "assistant-backend\assistant-backend.exe"
        if (!(Test-Path $backendExecutable)) {
                throw "Build completed but backend exe not found at: $backendExecutable"
    }

        Write-Host "[backend] Built: $backendExecutable"
        Write-Host "[backend] Packaging complete."
} finally {
    Pop-Location
}
