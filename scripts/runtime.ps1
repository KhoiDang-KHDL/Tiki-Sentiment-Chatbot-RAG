param([ValidateSet('config','build','start','stop','status','check','locust','password')][string]$Action='status')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')))
if ($Action -eq 'config') {
    & py -3.12 -m scripts.setup_runtime
    if ($LASTEXITCODE -ne 0) { throw 'Runtime configuration failed; Python 3.12 is required.' }
    return
}
if (-not (Test-Path -LiteralPath .env.runtime)) { throw 'Run scripts/runtime.ps1 config first.' }
function Invoke-Runtime {
    & docker compose --env-file .env.runtime -f docker-compose.runtime.yml @args
    if ($LASTEXITCODE -ne 0) { throw 'Runtime Compose command failed.' }
}
switch ($Action) {
    'build' { Invoke-Runtime build api ui }
    'start' { Invoke-Runtime up -d --wait --wait-timeout 600 }
    'stop' { Invoke-Runtime stop }
    'status' { Invoke-Runtime ps }
    'check' {
        Invoke-Runtime exec -T ui python -m runtime.chatbot.container_check
        Invoke-Runtime exec -T ui python -m runtime.chatbot.monitoring_check
    }
    'locust' { Invoke-Runtime --profile loadtest up -d api-loadtest locust }
    'password' {
        Get-Content .env.runtime | Where-Object { $_ -match '^GRAFANA_ADMIN_(USER|PASSWORD)=' }
    }
}
