# Waits for the N=8 Table-3 fill-in (run_n2_epochs.py) to finish, then runs
# cgae_dag on multi-site under the stated protocol (run_multisite_protocol.py).
# Launched detached on 2026-09-22; log in queue_cgae_dag.log.
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "queue_cgae_dag.log"
"[$(Get-Date -Format s)] waiting for run_n2_epochs.py to finish" | Out-File -Append -Encoding utf8 $log
while ($true) {
    $running = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*run_n2_epochs.py*" }
    if (-not $running) { break }
    Start-Sleep -Seconds 300
}
"[$(Get-Date -Format s)] N=8 run gone; launching cgae_dag on multi-site" | Out-File -Append -Encoding utf8 $log
cmd /c "python -u run_multisite_protocol.py 6 methods=cgae_dag >> suite_results_multisite_protocol_run.log 2>&1"
"[$(Get-Date -Format s)] cgae_dag run finished (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
