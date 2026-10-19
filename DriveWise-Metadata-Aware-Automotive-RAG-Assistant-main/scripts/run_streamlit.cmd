@echo off
setlocal
cd /d "%~dp0\.."
set "PYTHONDONTWRITEBYTECODE=1"
set "STREAMLIT_ARGS=run app.py --server.headless true --server.port 8501"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "CODEX_PY=C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if exist "%VENV_PY%" (
    call :try_python "%VENV_PY%"
    if not errorlevel 1 exit /b 0
)

if exist "%CODEX_PY%" (
    call :try_python "%CODEX_PY%"
    if not errorlevel 1 exit /b 0
)

where py >nul 2>nul
if not errorlevel 1 (
    call :try_python py
    if not errorlevel 1 exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
    call :try_python python
    if not errorlevel 1 exit /b 0
)

echo Could not start Streamlit. Install dependencies with: python -m pip install -r requirements.txt
exit /b 1

:try_python
set "PY_CMD=%~1"
"%PY_CMD%" -c "import streamlit" >nul 2>nul
if errorlevel 1 exit /b 1
"%PY_CMD%" -m streamlit %STREAMLIT_ARGS%
exit /b %ERRORLEVEL%
