@echo off
setlocal

cd /d "%~dp0"
title Auction Outcome Feasibility Predictor

echo ============================================================
echo  Auction Outcome Feasibility Predictor - Windows Launcher
echo ============================================================
echo.

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found in:
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
    %PYTHON_CMD% -m venv ".venv"
    if errorlevel 1 (
        echo ERROR: Could not create the Python environment.
        goto :failed
    )
)

echo Installing or confirming required dependencies...
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    goto :failed
)

echo.
echo Starting Streamlit. Press Ctrl+C in this window to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run "app.py" --browser.gatherUsageStats false --server.showEmailPrompt false
exit /b %ERRORLEVEL%

:failed
echo.
echo The application was not started.
pause
exit /b 1
