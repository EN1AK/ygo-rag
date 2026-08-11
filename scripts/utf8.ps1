$ErrorActionPreference = "Stop"

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8NoBom
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

git config --local core.quotepath false
git config --local i18n.commitEncoding utf-8
git config --local i18n.logOutputEncoding utf-8
git config --local gui.encoding utf-8

Write-Host "UTF-8 environment configured for this PowerShell session and local Git repository."
