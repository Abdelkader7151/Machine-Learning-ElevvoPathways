param(
    [string]$DataPath = "data/loan_approval_dataset.csv"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
if (!(Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    py -3 -m venv .venv
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"

Write-Host "Bootstrapping pip inside venv via ensurepip..."
& $pythonExe -m ensurepip --upgrade

Write-Host "Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing requirements..."
& $pythonExe -m pip install -r requirements.txt

$argsList = @("--data", $DataPath, "--outputs", "outputs")

Write-Host "Running training pipeline..."
& $pythonExe main.py @argsList

Write-Host "Done! Check the 'outputs' folder for generated artifacts."