#
# Jack Foundation - Deploy to Oracle Ampere
#
# This script deploys the Jack Foundation to an Oracle Ampere A1 instance.
#
# Usage:
#   .\deploy.ps1 -KeyFile "path\to\ssh-key.key" -Host "129.151.191.74"
#
# Requirements:
#   - SSH key for the Oracle instance
#   - PowerShell 5.1+ or PowerShell Core
#   - ssh and scp commands available (Windows 10+ has these)
#

param(
    [Parameter(Mandatory=$false)]
    [string]$KeyFile = "..\Oracle Instance 25gb Ampere\ssh-key-2026-02-08.key",

    [Parameter(Mandatory=$false)]
    [string]$Host = "129.151.191.74",

    [Parameter(Mandatory=$false)]
    [string]$User = "ubuntu",

    [Parameter(Mandatory=$false)]
    [string]$RemotePath = "/opt/jack",

    [switch]$SetupOnly,
    [switch]$CodeOnly,
    [switch]$RestartServices
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$JackRoot = Split-Path -Parent $ScriptDir
$JackFoundation = Join-Path $JackRoot "jack\foundation"
$JackServer = Join-Path $JackRoot "jack\server"

Write-Info "Jack Foundation Deployment"
Write-Info "=========================="
Write-Info "Host: $User@$Host"
Write-Info "Key: $KeyFile"
Write-Info "Remote: $RemotePath"
Write-Info ""

# Resolve key file path
$KeyFilePath = $KeyFile
if (-not [System.IO.Path]::IsPathRooted($KeyFile)) {
    $KeyFilePath = Join-Path $ScriptDir $KeyFile
}

if (-not (Test-Path $KeyFilePath)) {
    Write-Err "SSH key not found: $KeyFilePath"
    exit 1
}

Write-Info "Using SSH key: $KeyFilePath"

# SSH options
$SshOpts = @("-i", $KeyFilePath, "-o", "StrictHostKeyChecking=no")
$ScpOpts = @("-i", $KeyFilePath, "-o", "StrictHostKeyChecking=no", "-r")

function Invoke-SSH {
    param([string]$Command)
    $fullCmd = "ssh $($SshOpts -join ' ') $User@$Host `"$Command`""
    Write-Info "Running: $Command"
    Invoke-Expression $fullCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Err "SSH command failed"
        exit 1
    }
}

function Copy-ToRemote {
    param(
        [string]$Source,
        [string]$Dest
    )
    $fullCmd = "scp $($ScpOpts -join ' ') `"$Source`" `"$User@${Host}:$Dest`""
    Write-Info "Copying: $Source -> $Dest"
    Invoke-Expression $fullCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Err "SCP failed"
        exit 1
    }
}

# Test connection
Write-Info "Testing SSH connection..."
try {
    Invoke-SSH "echo 'Connection successful'"
} catch {
    Write-Err "Cannot connect to $Host"
    Write-Err "Make sure the instance is running and SSH key is correct"
    exit 1
}

# Run initial setup if requested
if ($SetupOnly) {
    Write-Info "Running initial setup..."

    # Copy setup script
    Copy-ToRemote (Join-Path $ScriptDir "setup-oracle.sh") "/tmp/setup-oracle.sh"

    # Run setup
    Invoke-SSH "chmod +x /tmp/setup-oracle.sh && sudo /tmp/setup-oracle.sh"

    Write-Info "Setup complete!"
    exit 0
}

# Create remote directories
Write-Info "Creating remote directories..."
Invoke-SSH "sudo mkdir -p $RemotePath/jack/foundation $RemotePath/jack/server"
Invoke-SSH "sudo chown -R $User:$User $RemotePath"

# Deploy code
if (-not $RestartServices) {
    Write-Info "Deploying Jack Foundation..."

    # Create temporary archive
    $TempZip = Join-Path $env:TEMP "jack-deploy.zip"
    if (Test-Path $TempZip) { Remove-Item $TempZip }

    # Compress jack package
    Write-Info "Creating deployment archive..."
    $JackPackage = Join-Path $JackRoot "jack"
    Compress-Archive -Path $JackPackage -DestinationPath $TempZip -Force

    # Copy to remote
    Copy-ToRemote $TempZip "/tmp/jack-deploy.zip"

    # Extract on remote
    Write-Info "Extracting on remote..."
    Invoke-SSH "cd $RemotePath && rm -rf jack && unzip -o /tmp/jack-deploy.zip && rm /tmp/jack-deploy.zip"

    # Copy requirements if exists
    $ReqFile = Join-Path $JackRoot "requirements.txt"
    if (Test-Path $ReqFile) {
        Copy-ToRemote $ReqFile "$RemotePath/requirements.txt"
        Invoke-SSH "cd $RemotePath && source venv/bin/activate && pip install -r requirements.txt"
    }

    # Cleanup
    Remove-Item $TempZip

    Write-Info "Code deployed successfully!"
}

# Restart services
if ($RestartServices -or (-not $SetupOnly -and -not $CodeOnly)) {
    Write-Info "Restarting services..."

    Invoke-SSH "sudo systemctl restart jack-server"
    Start-Sleep -Seconds 2

    # Check status
    Write-Info "Checking service status..."
    Invoke-SSH "sudo systemctl status jack-server --no-pager"

    # Test health endpoint
    Write-Info "Testing health endpoint..."
    Invoke-SSH "curl -s http://localhost:8000/health | python3 -m json.tool"
}

Write-Info ""
Write-Info "============================================"
Write-Info "  Deployment Complete!"
Write-Info "============================================"
Write-Info ""
Write-Info "API URL: http://$Host"
Write-Info ""
Write-Info "Test with:"
Write-Info "  curl http://$Host/health"
Write-Info ""
Write-Info "Create API key:"
Write-Info '  curl -X POST http://' + $Host + '/auth/token -H "Content-Type: application/json" -d "{\"user_id\":\"admin\",\"scopes\":[\"admin\"]}"'
Write-Info ""
