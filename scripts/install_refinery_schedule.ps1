<#
.SYNOPSIS
    Registers or updates the ThermoLab_LiveDataRefinery_6h Windows Scheduled Task
    to execute C:\Users\mcder\Cora\run_live_refinery_cron.ps1 every 6 hours.
    Use -DryRun to validate configuration without modifying Task Scheduler or requiring elevation.
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$TaskName = "ThermoLab_LiveDataRefinery_6h",
    [string]$RunnerScript = "C:\Users\mcder\Cora\run_live_refinery_cron.ps1"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RefineryScript = Join-Path $RepoRoot "scripts\run_live_refinery.py"
$LogFile = Join-Path $RepoRoot "logs\refinery_cron.log"

if (-not (Test-Path $RunnerScript)) {
    throw "Runner script not found: $RunnerScript"
}
if (-not (Test-Path $PythonExe)) {
    throw "Python venv interpreter not found: $PythonExe"
}
if (-not (Test-Path $RefineryScript)) {
    Write-Warning "Refinery runner script not yet present at $RefineryScript (expected before scheduled execution)."
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunnerScript`"" `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At "00:00" `
    -RepetitionInterval (New-TimeSpan -Hours 6)

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun:$false `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15) `
    -MultipleInstances IgnoreNew

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

if ($DryRun) {
    Write-Output "[DRY-RUN] Scheduled task configuration validated (no changes registered):"
    Write-Output "  TaskName           : $TaskName"
    Write-Output "  RunnerScript       : $RunnerScript"
    Write-Output "  WorkingDirectory   : $RepoRoot"
    Write-Output "  Action             : $($Action.Execute) $($Action.Arguments)"
    Write-Output "  RepetitionInterval : $($Trigger.Repetition.Interval) (6 hours)"
    Write-Output "  ExecutionTimeLimit : $($Settings.ExecutionTimeLimit)"
    Write-Output "  Principal          : $($Principal.UserId) (LogonType=$($Principal.LogonType), RunLevel=$($Principal.RunLevel))"
    Write-Output "  LogFile            : $LogFile"
    exit 0
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "ThermoLab 5-phase Live-Stream Data Refinery (--verify) running every 6 hours." `
    -Force | Out-Null

$Task = Get-ScheduledTask -TaskName $TaskName
Write-Output "Registered scheduled task '$($Task.TaskName)' (State: $($Task.State), Interval: PT6H) -> $RunnerScript"
