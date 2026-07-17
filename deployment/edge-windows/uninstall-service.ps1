[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Wrapper = Join-Path $PSScriptRoot "edge-runtime.exe"
if (-not (Test-Path $Wrapper -PathType Leaf)) {
    throw "Service wrapper not found: $Wrapper"
}
& $Wrapper stop
& $Wrapper uninstall
Write-Host "ConstructionSafetyEdge removed. Local configuration and models were preserved."
