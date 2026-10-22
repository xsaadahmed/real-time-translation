param(
    [switch]$Prepare,
    [switch]$FallbackOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.12 virtual environment missing. Run: py -3.12 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}
& $python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'This demo requires Python 3.12.' }

Push-Location $root
try {
    if ($Prepare) {
        if (-not $npm) { throw 'Node.js/npm is required to prepare the interface.' }
        & $python -m pip install -r requirements.txt
        & $python scripts/download_models.py --asr-model small --mt-backend marian --skip-tts --skip-camel-data
        Push-Location production-ui
        try { & $npm.Source ci; & $npm.Source run build } finally { Pop-Location }
        Write-Host 'Preparation complete. Test live mode before the exhibition.'
        exit 0
    }

    if ($FallbackOnly) {
        $env:RTT_FALLBACK_ONLY = '1'
        & $python run_demo.py --skip-install
    } else {
        & $python run_production.py --skip-install
    }
} finally {
    Pop-Location
}
