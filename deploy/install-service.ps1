<#
    Installs Rabbit Hole as a Windows service via NSSM.

    Run from an elevated PowerShell prompt:
        .\deploy\install-service.ps1 -Contact you@example.com

    Re-running is safe: the service is removed and recreated.
#>
[CmdletBinding()]
param(
    [string] $ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [string] $ServiceName = "RabbitHole",
    [int]    $Port = 8000,
    [string] $Contact = "email.address@gmail.com",

    # Cache stays on local disk. SQLite over SMB is a corruption risk, so this
    # must not point at a network share -- see deploy/README.md.
    [string] $CacheDir = "C:\ProgramData\rabbithole\cache",
    [string] $LogDir = "C:\ProgramData\rabbithole\logs",

    # A week is the app default. Wikipedia intros are stable; a month cuts
    # outbound API calls roughly fourfold for the same browsing.
    [int]    $CacheTtl = 2592000
)

$ErrorActionPreference = "Stop"

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "Run this from an elevated PowerShell prompt."
}

$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) {
    Write-Error "nssm not found on PATH. Install it first: winget install NSSM.NSSM"
}

$python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No venv at $python. Create it and 'pip install -e .' before installing the service."
}

New-Item -ItemType Directory -Force -Path $CacheDir, $LogDir | Out-Null

# Remove any previous install so this script is re-runnable.
if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing $ServiceName service..."
    & $nssm stop $ServiceName | Out-Null
    & $nssm remove $ServiceName confirm | Out-Null
    Start-Sleep -Milliseconds 500
}

Write-Host "Installing $ServiceName..."
& $nssm install $ServiceName $python "-m" "rabbithole" "--host" "0.0.0.0" "--port" "$Port"

& $nssm set $ServiceName AppDirectory $ProjectDir
& $nssm set $ServiceName DisplayName "Rabbit Hole"
& $nssm set $ServiceName Description "Wikipedia exploration canvas"
& $nssm set $ServiceName Start SERVICE_AUTO_START

& $nssm set $ServiceName AppEnvironmentExtra `
    "RABBITHOLE_CONTACT=$Contact" `
    "RABBITHOLE_CACHE_DIR=$CacheDir" `
    "RABBITHOLE_CACHE_TTL=$CacheTtl" `
    "PYTHONUNBUFFERED=1"

# Restart on crash, but back off so a config error doesn't spin.
& $nssm set $ServiceName AppExit Default Restart
& $nssm set $ServiceName AppRestartDelay 5000
& $nssm set $ServiceName AppThrottle 10000

& $nssm set $ServiceName AppStdout (Join-Path $LogDir "rabbithole.log")
& $nssm set $ServiceName AppStderr (Join-Path $LogDir "rabbithole.log")
& $nssm set $ServiceName AppRotateFiles 1
& $nssm set $ServiceName AppRotateBytes 10485760

$ruleName = "Rabbit Hole ($Port/tcp)"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    Write-Host "Opening port $Port on the private profile..."
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP `
        -LocalPort $Port -Action Allow -Profile Private | Out-Null
}

& $nssm start $ServiceName

Write-Host ""
Write-Host "Started. Reachable on the LAN at:"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    ForEach-Object { Write-Host "  http://$($_.IPAddress):$Port" }
Write-Host ""
Write-Host "Logs:  $LogDir\rabbithole.log"
Write-Host "Stop:  nssm stop $ServiceName"
