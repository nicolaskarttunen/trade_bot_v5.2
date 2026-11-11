[CmdletBinding()]
param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Expected audit-hardening-v1' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'ml_forward_pair.py'
$state = Join-Path $repo 'research_runs\ml-pair-20261111'
$sourceDir = Join-Path $repo 'research_runs\forward-20261110'
$envFile = Join-Path $env:USERPROFILE 'Downloads\trade_bot_v5.2\.env.v5'
$name = 'TradeBot-Simulation-MLPair-20261111'
$arg = '-u -X utf8 "{0}" update --state "{1}" --source "{2}" --env-file "{3}"' -f $script,$state,$sourceDir,$envFile
$action = New-ScheduledTaskAction -Execute $python -Argument $arg -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At '23:45'
$trigger.StartBoundary = '2026-12-09T21:45:00Z'
$trigger.EndBoundary = '2026-12-17T23:59:59Z'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$definition = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'After-close ML study; reads existing snapshots, no broker orders.'
if ($ValidateOnly) { Write-Host 'Task definition built. Not registered.'; return }
foreach ($p in @($python,$script,$envFile,(Join-Path $sourceDir 'ready.json'))) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing: $p" }
}
# This experiment needs the earlier collector; leave that task and directory alone.
$collector = Get-ScheduledTask -TaskName 'TradeBot-Simulation-Forward-20261110' -TaskPath '\' -ErrorAction Stop
if ($collector.State -eq 'Disabled') { throw 'The existing snapshot collector is disabled.' }
$existing = Get-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction SilentlyContinue
if ($existing -and ($existing.Actions.Count -ne 1 -or $existing.Actions[0].Execute -ne $python -or $existing.Actions[0].Arguments -ne $arg -or $existing.State -eq 'Disabled')) {
    throw "Existing task conflicts: $name"
}
& $python -u -X utf8 $script init --state $state --source $sourceDir
if ($LASTEXITCODE -ne 0) { throw 'Registration failed. Task was not installed.' }
if (-not $existing) { Register-ScheduledTask -TaskName $name -TaskPath '\' -InputObject $definition | Out-Null }
Write-Host "Scheduled task: $name"
(Get-ScheduledTaskInfo -TaskName $name -TaskPath '\') | Select-Object TaskName,NextRunTime
