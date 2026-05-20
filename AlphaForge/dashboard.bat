@echo off
title AlphaForge Dashboard
echo.
echo  AlphaForge Research Dashboard
echo  ==============================
echo  Starting on http://localhost:8501
echo.

cd /d "%~dp0"

:: Kill any existing Streamlit on port 8501 (only LISTENING state to avoid false matches)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: Launch dashboard in background
start "" /b py -m streamlit run harness/harness_dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

:: Wait for Streamlit to start, then open browser
timeout /t 4 /nobreak >nul
start "" "http://localhost:8501"

echo  Dashboard running at http://localhost:8501
echo  Press any key to stop the server and exit.
echo.
pause >nul

:: Kill Streamlit on exit
echo  Stopping server...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501 " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
:: Also catch ESTABLISHED connections from the same process
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
echo  Server stopped.
timeout /t 1 /nobreak >nul
