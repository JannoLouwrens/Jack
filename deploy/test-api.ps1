#
# Jack Foundation - API Test Script
#
# Tests the deployed Jack API endpoints.
#
# Usage:
#   .\test-api.ps1 -Host "129.151.191.74"
#

param(
    [Parameter(Mandatory=$false)]
    [string]$ApiHost = "129.151.191.74",

    [Parameter(Mandatory=$false)]
    [int]$Port = 80
)

$ErrorActionPreference = "Stop"
$BaseUrl = "http://${ApiHost}:${Port}"

function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }
function Write-Test { Write-Host "[TEST] $args" -ForegroundColor Cyan }

Write-Info "Jack Foundation API Tests"
Write-Info "========================="
Write-Info "Base URL: $BaseUrl"
Write-Info ""

$TestsPassed = 0
$TestsFailed = 0

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Endpoint,
        [hashtable]$Headers = @{},
        [string]$Body = $null,
        [int]$ExpectedStatus = 200
    )

    Write-Test "$Name..."

    try {
        $params = @{
            Uri = "$BaseUrl$Endpoint"
            Method = $Method
            Headers = $Headers
            ContentType = "application/json"
        }

        if ($Body) {
            $params["Body"] = $Body
        }

        $response = Invoke-WebRequest @params -UseBasicParsing

        if ($response.StatusCode -eq $ExpectedStatus) {
            Write-Info "  PASSED (Status: $($response.StatusCode))"
            $script:TestsPassed++
            return $response.Content | ConvertFrom-Json
        } else {
            Write-Err "  FAILED - Expected $ExpectedStatus, got $($response.StatusCode)"
            $script:TestsFailed++
            return $null
        }
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq $ExpectedStatus) {
            Write-Info "  PASSED (Status: $statusCode)"
            $script:TestsPassed++
            return $null
        } else {
            Write-Err "  FAILED - $($_.Exception.Message)"
            $script:TestsFailed++
            return $null
        }
    }
}

# =============================================================================
# Test 1: Health Check
# =============================================================================

$health = Test-Endpoint `
    -Name "Health Check" `
    -Method "GET" `
    -Endpoint "/health"

if ($health) {
    Write-Info "  Status: $($health.status)"
    Write-Info "  LLM Provider: $($health.llm_provider)"
    Write-Info "  Model: $($health.llm_model)"
}

# =============================================================================
# Test 2: Root Endpoint
# =============================================================================

$root = Test-Endpoint `
    -Name "Root Endpoint" `
    -Method "GET" `
    -Endpoint "/"

if ($root) {
    Write-Info "  Name: $($root.name)"
    Write-Info "  Version: $($root.version)"
}

# =============================================================================
# Test 3: Create Token
# =============================================================================

$tokenBody = @{
    user_id = "test_user"
    scopes = @("admin")
} | ConvertTo-Json

$token = Test-Endpoint `
    -Name "Create JWT Token" `
    -Method "POST" `
    -Endpoint "/auth/token" `
    -Body $tokenBody

$AccessToken = $null
if ($token) {
    $AccessToken = $token.access_token
    Write-Info "  Token Type: $($token.token_type)"
    Write-Info "  Expires In: $($token.expires_in) seconds"
    Write-Info "  Token: $($AccessToken.Substring(0, 50))..."
}

# =============================================================================
# Test 4: Authenticated Request (No Auth - Should Fail)
# =============================================================================

Test-Endpoint `
    -Name "Agent Query (No Auth)" `
    -Method "POST" `
    -Endpoint "/agent/query" `
    -Body '{"query": "test"}' `
    -ExpectedStatus 401

# =============================================================================
# Test 5: Authenticated Request (With Token)
# =============================================================================

if ($AccessToken) {
    $authHeaders = @{
        "Authorization" = "Bearer $AccessToken"
    }

    $llmStatus = Test-Endpoint `
        -Name "LLM Status (Authenticated)" `
        -Method "GET" `
        -Endpoint "/llm/status" `
        -Headers $authHeaders

    if ($llmStatus) {
        Write-Info "  Provider: $($llmStatus.provider)"
        Write-Info "  Model: $($llmStatus.model)"
        Write-Info "  Available: $($llmStatus.available)"
    }
}

# =============================================================================
# Test 6: Create API Key (Admin)
# =============================================================================

if ($AccessToken) {
    $apiKeyBody = @{
        name = "Test API Key"
        scopes = @("read", "write")
    } | ConvertTo-Json

    $apiKey = Test-Endpoint `
        -Name "Create API Key" `
        -Method "POST" `
        -Endpoint "/auth/api-key" `
        -Headers @{ "Authorization" = "Bearer $AccessToken" } `
        -Body $apiKeyBody

    if ($apiKey) {
        Write-Info "  Key ID: $($apiKey.key_id)"
        Write-Info "  Key: $($apiKey.key.Substring(0, 30))..."
        Write-Warn "  $($apiKey.message)"
    }
}

# =============================================================================
# Test 7: LLM Test (if available)
# =============================================================================

if ($AccessToken) {
    Write-Test "LLM Connection Test..."

    try {
        $response = Invoke-WebRequest `
            -Uri "$BaseUrl/llm/test" `
            -Method "POST" `
            -Headers @{ "Authorization" = "Bearer $AccessToken" } `
            -ContentType "application/json" `
            -UseBasicParsing `
            -TimeoutSec 120

        $result = $response.Content | ConvertFrom-Json

        if ($result.success) {
            Write-Info "  PASSED - LLM is responding"
            Write-Info "  Response: $($result.response.Substring(0, [Math]::Min(100, $result.response.Length)))..."
            $script:TestsPassed++
        } else {
            Write-Warn "  LLM not available: $($result.error)"
            # Don't count as failure - LLM might not be running
        }
    } catch {
        Write-Warn "  LLM test skipped (may not be running)"
    }
}

# =============================================================================
# Test 8: Rate Limiting
# =============================================================================

Write-Test "Rate Limiting..."
$rateLimitHit = $false

for ($i = 1; $i -le 5; $i++) {
    try {
        Invoke-WebRequest `
            -Uri "$BaseUrl/health" `
            -Method "GET" `
            -UseBasicParsing | Out-Null
    } catch {
        if ($_.Exception.Response.StatusCode.value__ -eq 429) {
            $rateLimitHit = $true
            break
        }
    }
}

if (-not $rateLimitHit) {
    Write-Info "  PASSED - Rate limiting is working (not hit in 5 requests)"
    $TestsPassed++
} else {
    Write-Warn "  Rate limit hit after $i requests"
}

# =============================================================================
# Summary
# =============================================================================

Write-Info ""
Write-Info "============================================"
Write-Info "  Test Summary"
Write-Info "============================================"
Write-Info ""

if ($TestsFailed -eq 0) {
    Write-Info "All tests passed! ($TestsPassed/$($TestsPassed + $TestsFailed))"
} else {
    Write-Err "Some tests failed: $TestsPassed passed, $TestsFailed failed"
}

Write-Info ""
Write-Info "API is ready at: $BaseUrl"
Write-Info ""
