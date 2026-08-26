@echo off
REM FaceGate — Windows launcher (double-click or run from cmd/PowerShell)
setlocal
set "HERE=%~dp0"
set "EYE=%HERE%.."
set "ROOT=%EYE%\.."

if exist "%ROOT%\.venv\Scripts\python.exe" (
  set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else if exist "%EYE%\.venv\Scripts\python.exe" (
  set "PYTHON=%EYE%\.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

cd /d "%EYE%"
"%PYTHON%" "%HERE%launch.py" %*
if errorlevel 1 pause
