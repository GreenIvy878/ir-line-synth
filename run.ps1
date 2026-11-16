# Launch the editor and open it in the browser.
# Usage: .\run.ps1 -Data <folder or data.yaml> [-Port 8009]
# Without -Data the dataset is taken from the SCENE_EDITOR_DATA environment variable, so a
# machine can keep its own source once:  setx SCENE_EDITOR_DATA "C:\path\to\dataset"
# A relative -Data is taken relative to this folder; the editor prints the dataset it reads.
param(
    [int]$Port = 8009,
    [string]$Data = $env:SCENE_EDITOR_DATA
)

if (-not $Data) {
    Write-Error "Name the source dataset: .\run.ps1 -Data <folder or data.yaml>, or set SCENE_EDITOR_DATA."
    exit 1
}

$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$dataPath = if ([System.IO.Path]::IsPathRooted($Data)) { $Data } else { Join-Path $root $Data }

if (-not (Test-Path $python)) {
    py -3.12 -m venv (Join-Path $root ".venv")
}
& $python -c "import scene_editor" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $python -m pip install -e $root
}

Start-Job -ArgumentList $Port -ScriptBlock {
    param($p)
    Start-Sleep -Seconds 2
    Start-Process "http://127.0.0.1:$p/"
} | Out-Null

& $python -m scene_editor serve --data $dataPath --port $Port
