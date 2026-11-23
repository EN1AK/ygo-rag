$ErrorActionPreference = 'Stop'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = $utf8
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8

$repo = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $repo 'build\reference-rag-corpora'
[System.IO.Directory]::CreateDirectory($logDirectory) | Out-Null
$logPath = Join-Path $logDirectory 'scheduled-refresh.log'

function Write-RefreshLog([string] $Message) {
    $stamp = [DateTimeOffset]::Now.ToString('o')
    [System.IO.File]::AppendAllText($logPath, "[$stamp] $Message" + [Environment]::NewLine, $utf8)
}

Write-RefreshLog 'Starting upstream OCG corpus refresh.'
$exitCode = 1
try {
    # Keep native stderr in the task log while using WSL's actual exit code.
    $ErrorActionPreference = 'Continue'
    & "$env:SystemRoot\System32\wsl.exe" --exec  `
        /mnt/d/workspace/rag/build/native-venv/bin/python -X utf8  `
        /mnt/d/workspace/rag/scripts/sync_upstream_ocg_corpus.py refresh 2>&1 |
        ForEach-Object { Write-RefreshLog $_.ToString().Replace([string][char]0, '') }
    $exitCode = $LASTEXITCODE
} catch {
    Write-RefreshLog $_.Exception.Message
}
Write-RefreshLog "Upstream OCG corpus refresh exited with code $exitCode."
exit $exitCode
