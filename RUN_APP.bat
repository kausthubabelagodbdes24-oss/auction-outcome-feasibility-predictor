@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
title Auction Outcome Feasibility Predictor
set "SERVER_HEADLESS=false"
if /i "%~1"=="--headless" set "SERVER_HEADLESS=true"

echo ============================================================
echo  Auction Outcome Feasibility Predictor - Windows Launcher
echo ============================================================
echo.

if exist "requirements-app.txt" (
    set "DEPENDENCY_FILE=requirements-app.txt"
) else if exist "requirements.txt" (
    set "DEPENDENCY_FILE=requirements.txt"
) else (
    echo ERROR: No dependency file was found in:
    echo        %CD%
    goto :failed
)

if defined AUCTION_MODEL_BUNDLE (
    if not exist "%AUCTION_MODEL_BUNDLE%" (
        echo ERROR: AUCTION_MODEL_BUNDLE points to a file that does not exist:
        echo        %AUCTION_MODEL_BUNDLE%
        goto :failed
    )
) else (
    if not exist "auction_model_bundle.zip" (
        echo ERROR: auction_model_bundle.zip was not found in:
        echo        %CD%
        echo.
        echo Run the Google Colab notebook, download the bundle, and place it
        echo beside this RUN_APP.bat file.
        goto :failed
    )
)

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo ERROR: Python 3 was not found.
            echo Install Python 3, enable "Add Python to PATH", and try again.
            goto :failed
        )
        set "PYTHON_CMD=python"
    )

    echo Creating the local Python environment...
    !PYTHON_CMD! -m venv ".venv"
    if errorlevel 1 (
        echo ERROR: Could not create the Python environment.
        goto :failed
    )
)

if not exist ".venv\.auction-runtime-v1.0.0" (
    echo Installing the compact application runtime...
    ".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check --no-cache-dir -r "%DEPENDENCY_FILE%"
    if errorlevel 1 (
        echo ERROR: Dependency installation failed.
        goto :failed
    )
    > ".venv\.auction-runtime-v1.0.0" echo ready
    if errorlevel 1 (
        echo ERROR: Could not finalize the runtime environment.
        goto :failed
    )
) else (
    echo Reusing the installed application runtime.
)

echo.
echo Starting Streamlit. Press Ctrl+C in this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run "app.py" --server.headless %SERVER_HEADLESS% --browser.gatherUsageStats false --server.showEmailPrompt false
exit /b %ERRORLEVEL%

:failed
echo.
echo The application was not started.
pause
exit /b 1
