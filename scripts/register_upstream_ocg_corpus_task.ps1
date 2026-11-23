param([switch] $RunNow)

$ErrorActionPreference = 'Stop'
$name = 'OCG-Upstream-Corpus-Refresh'
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    throw "Scheduled task '$name' already exists."
}
$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repo 'scripts\run_upstream_ocg_corpus_refresh.ps1'
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "' + $runner + '"') -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At 9am
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Refresh verified public upstream OCG corpus snapshots each Monday.' | Select-Object TaskName, State
if ($RunNow) { Start-ScheduledTask -TaskName $name }
