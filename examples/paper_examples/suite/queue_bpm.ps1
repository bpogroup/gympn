# BPM conference-paper compute queue. Runs run_bpm.py jobs back to back on
# 4 workers; each is resumable by cell file, so the queue can be killed and
# relaunched at any time. Logs: bpm_<env>_n<N>.log (UTF-8) + queue_bpm.log.
#
#   .\queue_bpm.ps1                 # full run: N=1 and N=8, 20 seeds
#   .\queue_bpm.ps1 -Seeds 10 -Pilot  # pilot: N=1 only, 10 seeds
param([int]$Seeds = 20, [switch]$Pilot)
$ErrorActionPreference = 'Continue'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
Set-Location $PSScriptRoot
$py = "C:\Users\lobia\PycharmProjects\gympn\.venv\Scripts\python.exe"
$jobs = @(
    @{ env = 'next_activity'; N = 1 },
    @{ env = 'rework';        N = 1 }
)
if (-not $Pilot) {
    $jobs += @{ env = 'next_activity'; N = 8 }
    $jobs += @{ env = 'rework';        N = 8 }
}
foreach ($j in $jobs) {
    $log = "bpm_$($j.env)_n$($j.N).log"
    "[$(Get-Date -Format s)] start env=$($j.env) N=$($j.N) seeds=$Seeds" | Out-File -Append queue_bpm.log
    & $py -u run_bpm.py "env=$($j.env)" "N=$($j.N)" "seeds=$Seeds" 4 2>&1 | Out-File -Append $log
    "[$(Get-Date -Format s)] done  env=$($j.env) N=$($j.N) exit=$LASTEXITCODE" | Out-File -Append queue_bpm.log
}
"[$(Get-Date -Format s)] queue finished" | Out-File -Append queue_bpm.log
