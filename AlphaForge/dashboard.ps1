# AlphaForge Dashboard Launcher
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "  AlphaForge Research Dashboard" -ForegroundColor Cyan
Write-Host "  ==============================" -ForegroundColor DarkCyan
Write-Host "  Starting at http://localhost:8501" -ForegroundColor Gray
Write-Host ""

# Kill any existing Streamlit on port 8501
# NOTE: $pid is a reserved variable in PowerShell (current PID) — use $serverPid
$existing = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($existing) {
    $serverPid = ($existing | Select-Object -First 1).OwningProcess
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    Write-Host "  Stopped existing server (PID $serverPid)" -ForegroundColor Yellow
    Start-Sleep 1
}

# Start Streamlit in background (hidden window)
$proc = Start-Process -FilePath "py" `
    -ArgumentList "-m streamlit run harness/harness_dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false" `
    -PassThru -WindowStyle Hidden

if (-not $proc) {
    Write-Host "  ERROR: Failed to start server. Is Python installed?" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

Write-Host "  Server started (PID $($proc.Id))" -ForegroundColor Green
Write-Host "  Waiting for startup..." -ForegroundColor Gray
Start-Sleep 4

# Open browser
Start-Process "http://localhost:8501"
Write-Host "  Browser opened at http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "  Press Enter to stop the dashboard server." -ForegroundColor Gray
Read-Host

# Graceful shutdown: kill the process tree so streamlit subprocesses also stop
Write-Host "  Stopping server..." -ForegroundColor Yellow
try {
    # Kill the process tree (py.exe + any spawned streamlit workers)
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $proc.Id }
    foreach ($child in $children) {
        Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
} catch {}

# Belt-and-braces: kill anything still holding port 8501
$remaining = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue
if ($remaining) {
    $remaining | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "  Server stopped." -ForegroundColor Yellow
