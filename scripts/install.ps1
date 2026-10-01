param([switch]$Full)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw '需要先安装 Python 3.12。' }
    }
    $dependencyFile = if ($Full) { 'requirements.txt' } else { 'requirements-harness.txt' }
    & '.venv/Scripts/python.exe' -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'pip 初始化失败。' }
    & '.venv/Scripts/python.exe' -m pip install -r $dependencyFile
    if ($LASTEXITCODE -ne 0) { throw '依赖安装失败。' }
    Write-Host '安装完成。运行 .venv/Scripts/python.exe -m app.launcher'
} finally { Pop-Location }
