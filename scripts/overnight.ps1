# Overnight pipeline: wait for the GPS download, verify, then run everything. Logs in data\logs\.
#   powershell -ExecutionPolicy Bypass -File scripts\overnight.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:BUSPULSE_DATA = ""
Remove-Item Env:\BUSPULSE_DATA -ErrorAction SilentlyContinue
$logs = "data\logs"; New-Item -ItemType Directory -Force $logs | Out-Null
$arch = "data\raw\delhi-dtc-bus-gtfs-rt-2026-10.7z"
$url = "https://pub-091bd9e794bb4877b10b29e6df80f065.r2.dev/2026-10/delhi-dtc-bus-gtfs-rt-2026-10.7z"
$want = "703bd6ea5237883e1ad8f094a661d365edc74b7555cf83ce4c554408a79c16c9"
function Log($m) { $l = "$(Get-Date -Format 'HH:mm:ss') $m"; Write-Host $l; Add-Content "$logs\overnight.log" $l }

# 1. download (resume until the checksum matches; up to 20 attempts)
for ($i = 0; $i -lt 20; $i++) {
    while (Get-Process curl -ErrorAction SilentlyContinue) { Start-Sleep 60; Log ("download {0:N2} GB" -f ((Get-Item $arch).Length / 1GB)) }
    $h = (certutil -hashfile $arch SHA256)[1].Replace(" ", "")
    if ($h -eq $want) { Log "checksum OK"; break }
    Log "checksum mismatch / incomplete ($("{0:N2}" -f ((Get-Item $arch).Length/1GB)) GB) - resuming"
    & curl.exe -s -C - -o $arch $url
}
& "C:\Program Files\7-Zip\7z.exe" l -slt $arch | Select-String "^Solid|^Path = .*\.db" | Out-File "$logs\archive_listing.txt"
Log "archive listing saved"

# 2. GTFS must be present
if (-not (Get-ChildItem data\gtfs -Recurse -Include stops.txt, *.zip -ErrorAction SilentlyContinue)) {
    Log "STOP: no GTFS in data\gtfs - download it, then re-run this script"; exit 1
}
python scripts\00_inspect.py --skip-gps *> "$logs\00_inspect.txt"; Log "00 inspect done"

# 3. pipeline (each step logged; extraction is resumable if interrupted)
python scripts\01_gps_extract.py --workers 2 *> "$logs\01_extract.txt"; Log "01 extract done"
python scripts\02_build_traveltime.py *> "$logs\02_model.txt"; Log "02 model done"
python scripts\03_tickets_crowding.py *> "$logs\03_crowding.txt"; Log "03 crowding done"
python scripts\07_insights.py *> "$logs\07_insights.txt"; Log "07 insights done"
python scripts\04_validate.py *> "$logs\04_validate.txt"; Log "04 validation done"
python scripts\08_data_quality.py *> "$logs\08_quality.txt"; Log "08 data quality done"
Log "ALL DONE"
