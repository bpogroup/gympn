# BPM conference-paper compute queue (2026-09-28). Runs the four run_bpm.py
# jobs back to back on 4 workers; each is resumable by cell file, so the
# queue can be killed and relaunched at any time. Logs: bpm_<env>_n<N>.log.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = "C:\Users\lobia\PycharmProjects\gympn\.venv\Scripts\python.exe"
$jobs = @(
    @{ env = 'next_activity'; N = 1 },
    @{ env = 'rework';        N = 1 },
    @{ env = 'next_activity'; N = 8 },
    @{ env = 'rework';        N = 8 }
)
foreach ($j in $jobs) {
    $log = "bpm_$($j.env)_n$($j.N).log"
    "[$(Get-Date -Format s)] start env=$($j.env) N=$($j.N)" | Out-File -Append queue_bpm.log -Encoding utf8
    & $py -u run_bpm.py "env=$($j.env)" "N=$($j.N)" 4 *>> $log
    "[$(Get-Date -Format s)] done  env=$($j.env) N=$($j.N) exit=$LASTEXITCODE" | Out-File -Append queue_bpm.log -Encoding utf8
}
"[$(Get-Date -Format s)] queue finished" | Out-File -Append queue_bpm.log -Encoding utf8
