@echo off
setlocal
cd /d %~dp0\..

set PYTHON_BIN=python
if exist ".venv\Scripts\python.exe" (
  set PYTHON_BIN=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
  set PYTHON_BIN=venv\Scripts\python.exe
)

%PYTHON_BIN% -m cot_bias fetch --update
if errorlevel 1 goto :error

%PYTHON_BIN% -m cot_bias compute --out outputs
if errorlevel 1 goto :error

%PYTHON_BIN% -m cot_bias dashboard --out outputs
if errorlevel 1 goto :error

%PYTHON_BIN% scripts\generate_recent_dashboards.py --months 4 --out outputs
if errorlevel 1 goto :error

echo Done. Latest output in .\outputs\ and last 4 months in .\outputs\YYYY-MM-DD\
endlocal
exit /b 0

:error
echo Script failed.
endlocal
exit /b 1
