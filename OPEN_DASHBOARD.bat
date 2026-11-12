@echo off
setlocal

cd /d "%~dp0"
title Auction Outcome Feasibility Predictor - Dashboard

echo ============================================================
echo  Auction Outcome Feasibility Predictor - Astryx Dashboard
echo ============================================================
echo.

set "MAIN_REPO=%USERPROFILE%\Documents\Auction Verification"

if defined AUCTION_MODEL_BUNDLE (
    if not exist "%AUCTION_MODEL_BUNDLE%" (
        echo ERROR: AUCTION_MODEL_BUNDLE points to a missing file:
        echo        %AUCTION_MODEL_BUNDLE%
        goto :failed
    )
) else if exist "auction_model_bundle.zip" (
    set "AUCTION_MODEL_BUNDLE=%CD%\auction_model_bundle.zip"
) else if exist "%MAIN_REPO%\auction_model_bundle.zip" (
    set "AUCTION_MODEL_BUNDLE=%MAIN_REPO%\auction_model_bundle.zip"
) else (
    echo ERROR: auction_model_bundle.zip was not found in:
    echo        %CD%
    echo        or %MAIN_REPO%
    echo.
    echo Place auction_model_bundle.zip in the repository root or set AUCTION_MODEL_BUNDLE.
    goto :failed
)

set "VENV_PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
) else if exist "%MAIN_REPO%\.venv\Scripts\python.exe" (
    set "VENV_PYTHON=%MAIN_REPO%\.venv\Scripts\python.exe"
)

if not defined VENV_PYTHON (
    if not exist "requirements.txt" (
        echo ERROR: requirements.txt was not found in %CD%
        goto :failed
    )
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3 was not found on PATH.
            goto :failed
        )
        set "PYTHON_CMD=python"
    )
    echo Creating local Python environment...
    %PYTHON_CMD% -m venv ".venv"
    if errorlevel 1 goto :failed
    set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
    echo Installing dependencies from requirements.txt...
    "%VENV_PYTHON%" -m pip install --quiet --disable-pip-version-check -r "requirements.txt"
    if errorlevel 1 goto :failed
)

echo Bundle : %AUCTION_MODEL_BUNDLE%
echo Python : %VENV_PYTHON%
echo.
if /i "%~1"=="--dry-run" exit /b 0

echo Opening Streamlit dashboard in your browser...
echo Press Ctrl+C in this window to stop the server.
echo.

"%VENV_PYTHON%" -m streamlit run "app.py" --server.headless false --browser.gatherUsageStats false --server.showEmailPrompt false
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo The dashboard could not be started.
pause
exit /b 1
