<#
.SYNOPSIS
Launch a private 48-hour OpenAI-compatible API sandbox endpoint with bearer-token auth.

.EXAMPLE
.\tools\tensor-compressor\deploy\start_sandbox.ps1 -Model "qwen-qwen3.8-27b-compressed" -Port 8080
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [int]$Port = 8080,

    [ValidateSet("auto", "llama-server", "ollama")]
    [string]$Backend = "auto",

    [double]$Hours = 48,

    [string]$Token = "",

    [switch]$ExposeTunnel,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Token)) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $Token = "nxs_" + ([Convert]::ToBase64String($bytes) -replace '\+', '-' -replace '/', '_' -replace '=', '')
}

$expiresUtc = (Get-Date).ToUniversalTime().AddHours($Hours).ToString("yyyy-MM-ddTHH:mm:ssZ")
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
$venvPy = Join-Path $repoRoot "tools\tensor-compressor\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { $venvPy = "python" }

$resolvedBackend = $Backend
if ($resolvedBackend -eq "auto") {
    $hasLlama = $null -ne (Get-Command "llama-server" -ErrorAction SilentlyContinue)
    if ($hasLlama -and ($Model -match '\.gguf$') -and (Test-Path $Model)) {
        $resolvedBackend = "llama-server"
    } else {
        $resolvedBackend = "ollama"
    }
}

Write-Host "========================================================================"
Write-Host "NEXUS 48-HOUR PROSPECT SANDBOX"
Write-Host "Model:       $Model"
Write-Host "Backend:     $resolvedBackend"
Write-Host "Endpoint:    http://localhost:$Port/v1/chat/completions"
Write-Host "Bearer:      $Token"
Write-Host "Expires UTC: $expiresUtc ($Hours hours)"
Write-Host "========================================================================"
Write-Host ""
Write-Host "# Test with curl:"
Write-Host "curl http://localhost:$Port/v1/chat/completions -H `"Authorization: Bearer $Token`" -H `"Content-Type: application/json`" -d '{`"model`":`"$Model`",`"messages`":[{`"role`":`"user`",`"content`":`"ping`"}]}'"
Write-Host ""

if ($DryRun) {
    Write-Host "[DryRun] Validation complete; no server process started."
    exit 0
}

$tunnelProc = $null
if ($ExposeTunnel) {
    $cf = Get-Command "cloudflared" -ErrorAction SilentlyContinue
    if ($cf) {
        Write-Host "[Tunnel] Starting cloudflared quick tunnel -> http://127.0.0.1:$Port ..."
        $tunnelProc = Start-Process -FilePath $cf.Source -ArgumentList @("tunnel", "--url", "http://127.0.0.1:$Port") -PassThru -NoNewWindow
    } else {
        Write-Warning "cloudflared not found on PATH; running on localhost:$Port only."
    }
}

try {
    if ($resolvedBackend -eq "llama-server") {
        $llamaExe = (Get-Command "llama-server" -ErrorAction Stop).Source
        Write-Host "[Sandbox] Starting llama-server on port $Port with API key auth..."
        $proc = Start-Process -FilePath $llamaExe -ArgumentList @(
            "--model", $Model,
            "--host", "0.0.0.0",
            "--port", "$Port",
            "--api-key", $Token
        ) -PassThru -NoNewWindow
        $timeoutMs = [int]([Math]::Min([int]::MaxValue, $Hours * 3600 * 1000))
        if (-not $proc.WaitForExit($timeoutMs)) {
            Write-Host "[Sandbox] $Hours-hour TTL reached; stopping llama-server..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    } else {
        $ollamaExe = (Get-Command "ollama" -ErrorAction Stop).Source
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2 | Out-Null
        } catch {
            Write-Host "[Sandbox] Starting Ollama daemon..."
            Start-Process -FilePath $ollamaExe -ArgumentList @("serve") -WindowStyle Hidden | Out-Null
            Start-Sleep -Seconds 3
        }
        $proxyPy = Join-Path $scriptDir "sandbox_auth_proxy.py"
        & $venvPy $proxyPy --port $Port --upstream "http://127.0.0.1:11434" --token $Token --ttl-hours $Hours
    }
} finally {
    if ($tunnelProc -and (-not $tunnelProc.HasExited)) {
        Stop-Process -Id $tunnelProc.Id -Force -ErrorAction SilentlyContinue
    }
}
