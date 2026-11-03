@echo off
title Nexus AI Solutions - Autonomous Funnel & Command Center
cd /d "%~dp0.."

echo ============================================================================
echo   NEXUS AI SOLUTIONS - ONE-CLICK LAUNCHER
echo ============================================================================
echo   [1/3] Running Autonomous NYC Contractor Hunter (hunter_agent.py)...
".\.venv\Scripts\python.exe" nexus_ai_funnel\hunter_agent.py

echo.
echo   [2/3] Running Community Intent Monitor (intent_monitor.py)...
".\.venv\Scripts\python.exe" nexus_ai_funnel\intent_monitor.py --no-live

echo.
echo   [3/3] Starting Unified Webhook Server & Opening Visual Command Center...
start "" "http://localhost:8000/dashboard"
".\.venv\Scripts\python.exe" nexus_ai_funnel\webhook_server.py --port 8000
