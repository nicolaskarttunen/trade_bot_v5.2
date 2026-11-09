# Installs a user-level scheduled task after freezing the verified research model.
[CmdletBinding()]
param([switch]$ValidateOnly)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if ((git branch --show-current) -ne 'audit-hardening-v1') { throw 'Expected audit-hardening-v1' }
$python = Join-Path $repo '.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'watch_forward.py'
$state = Join-Path $repo 'research_runs\forward-20261110'
$old = Join-Path $env:USERPROFILE 'Downloads\trade_bot_v5.2'
$envFile = Join-Path $old '.env.v5'
$reference = Join-Path $repo 'research_runs\jarjestys-20261109-130806\reference'
$name = 'TradeBot-Simulation-Forward-20261110'
# Build the action without putting any credential values in task arguments.
$arg = '-u -X utf8 "{0}" update --state "{1}" --env-file "{2}"' -f $script,$state,$envFile
$action = New-ScheduledTaskAction -Execute $python -Argument $arg -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At '23:45'
$trigger.StartBoundary = '2026-11-10T21:45:00Z'
$trigger.EndBoundary = '2026-12-10T23:59:59Z'
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 6)
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$definition = New-ScheduledTask -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Read-only SIP history; four simulated policies, no broker orders.'
if ($ValidateOnly) { Write-Host 'Task definition built. No task registered.'; return }
foreach ($path in @($python,$script,$envFile,$reference)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing: $path" } }
# Never overwrite an unrelated task, even if its name matches.
$existing = Get-ScheduledTask -TaskName $name -TaskPath '\' -ErrorAction SilentlyContinue
if ($existing -and ($existing.Actions.Count -ne 1 -or $existing.Actions[0].Execute -ne $python -or $existing.Actions[0].Arguments -ne $arg)) {
    throw "Another task uses the name $name"
}
& $python -u -X utf8 $script init --state $state --train-data (Join-Path $old 'data_sip') --reference $reference
if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Task was not installed.' }
if (-not $existing) {
    Register-ScheduledTask -TaskName $name -TaskPath '\' -InputObject $definition | Out-Null
}
Write-Host "Scheduled task: $name"
& $python -u -X utf8 $script update --state $state --env-file $envFile
if ($LASTEXITCODE -ne 0) { throw 'First check failed. See last_attempt.json in the state directory.' }
Write-Host "State: $state"
(Get-ScheduledTaskInfo -TaskName $name -TaskPath '\') | Select-Object TaskName,NextRunTime
