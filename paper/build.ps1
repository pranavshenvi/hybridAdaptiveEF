# Build paper\main.pdf with MiKTeX (installed per-user via winget). Usage, from the paper folder:
#   powershell -ExecutionPolicy Bypass -File build.ps1
$bin = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"
Set-Location $PSScriptRoot
& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
& "$bin\bibtex.exe" main | Out-Null
& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex | Out-Null
& "$bin\pdflatex.exe" -interaction=nonstopmode main.tex | Select-Object -Last 2
Select-String -Path main.log -Pattern "LaTeX Error|Undefined control|undefined on input" | ForEach-Object { $_.Line }
Start-Process main.pdf
