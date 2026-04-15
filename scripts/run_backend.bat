@echo off
REM OpenSAMAnnotator - Backend Startup Script (Windows)
setlocal enabledelayedexpansion

echo.
echo ========================================
echo   OpenSAMAnnotator - Backend Setup
echo ========================================
echo.

REM Navigate to backend directory
cd /d "%~dp0..\backend"

REM Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo       Python %PYVER% found

REM Check if virtual environment exists
echo [2/5] Setting up virtual environment...
if not exist "venv" (
    echo       Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo       Virtual environment created
) else (
    echo       Virtual environment exists
)

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo       Activated

REM Install/update dependencies
echo [4/5] Installing dependencies...
echo       This may take a few minutes on first run...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo       Some packages may have failed, attempting without quiet mode...
    pip install -r requirements.txt
)
echo       Dependencies installed

REM Start the server
echo [5/5] Starting FastAPI server...
echo.
echo ========================================
echo   Backend server starting!
echo ========================================
echo.
echo   API URL:    http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo   Health:     http://localhost:8000/api/health
echo.
echo   IMPORTANT: Now open a NEW terminal and run:
echo   cd .. ^&^& npm run dev
echo   (or: cd .. ^&^& pnpm dev)
echo.
echo   Then open http://localhost:3000 in your browser
echo.
echo ========================================
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

uvicorn main:app --reload --host 0.0.0.0 --port 8000
