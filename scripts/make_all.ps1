# Rebuild every model table and every report from the extracted GPS + tickets (after 01_gps_extract).
#   powershell -ExecutionPolicy Bypass -File scripts\make_all.ps1            (logs in data\logs\)
# Needs a free port 8790 for the stress test (a temporary server is started and stopped).
$ErrorActionPreference = "Continue"   # Python warnings on stderr must not abort a step; exit codes decide
Set-Location (Split-Path $PSScriptRoot -Parent)
Remove-Item Env:\BUSPULSE_DATA -ErrorAction SilentlyContinue
Remove-Item Env:\BUSPULSE_DOCS -ErrorAction SilentlyContinue
$env:PYTHONUNBUFFERED = "1"
$logs = "data\logs"; New-Item -ItemType Directory -Force $logs | Out-Null
function Step($name, $cmd) {
    $t = Get-Date
    Write-Host "== $name"
    & python @cmd *> "$logs\$name.txt"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name (see $logs\$name.txt)"; exit 1 }
    Write-Host ("   done in {0:N0} s" -f ((Get-Date) - $t).TotalSeconds)
}
Step "02_model"      @("scripts\02_build_traveltime.py")
Step "03_crowding"   @("scripts\03_tickets_crowding.py")
Step "07_insights"   @("scripts\07_insights.py")
Step "04_validate"   @("scripts\04_validate.py")
Step "11_rides"      @("scripts\11_validate_rides.py", "--n", "420")
Step "08_quality"    @("scripts\08_data_quality.py")
Step "09_bundle"     @("scripts\09_bundle.py")
if (Test-Path data\google_benchmark.csv) { Step "05_google" @("scripts\05_google_compare.py") }
# stress test against a server running from the deploy bundle (what Render runs)
$env:BUSPULSE_DATA = "bundle"
$srv = Start-Process -FilePath python -ArgumentList "run.py", "--port", "8790" -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput "$logs\server_8790.txt" -RedirectStandardError "$logs\server_8790_err.txt"
Remove-Item Env:\BUSPULSE_DATA
Start-Sleep 8
Step "06_stress"     @("scripts\06_loadtest.py", "--url", "http://127.0.0.1:8790")
Stop-Process -Id $srv.Id -Force
Write-Host "ALL DONE - reports in docs\"
