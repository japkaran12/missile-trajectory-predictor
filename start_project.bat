@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: 1) Set repo root to the folder where this .bat is located
cd /d "%~dp0"
set "REPO_DIR=%CD%"

echo.
echo ==============================
echo Missile Trajectory Predictor - One-Click Launcher
echo Repo: %REPO_DIR%
echo ==============================
echo.

:: 2) Check Python
where python >nul 2>&1
if ERRORLEVEL 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.8+ and re-run.
    pause
    exit /b 1
)

:: 3) Create venv if it doesn't exist
if not exist "%REPO_DIR%\venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%REPO_DIR%\venv"
    if ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

set "VENV_PY=%REPO_DIR%\venv\Scripts\python.exe"

:: 4) Upgrade pip and install requirements
echo Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel >nul

if exist "%REPO_DIR%\requirements.txt" (
    echo Installing dependencies from requirements.txt ...
    "%VENV_PY%" -m pip install -r "%REPO_DIR%\requirements.txt"
) else (
    echo requirements.txt not found — installing default dependencies...
    "%VENV_PY%" -m pip install numpy pandas scikit-learn matplotlib joblib streamlit
)

if ERRORLEVEL 1 (
    echo [ERROR] pip install failed. Check internet connection and retry.
    pause
    exit /b 1
)

:: 5) Ensure data folder exists
if not exist "%REPO_DIR%\data" mkdir "%REPO_DIR%\data"

:: 6) Train model if missing
set "MODEL_PATH=%REPO_DIR%\models\rf_missile_range.joblib"
if not exist "%MODEL_PATH%" (
    echo Trained model not found. Training model now (this may take a few minutes)...
    "%VENV_PY%" "%REPO_DIR%\scripts\missile_ml_simple.py"
    if ERRORLEVEL 1 (
        echo [ERROR] Training script failed.
        pause
        exit /b 1
    )
) else (
    echo Model already exists at %MODEL_PATH%
)

:: 7) Optional evaluation step (comment/uncomment if you want)
:: echo Running evaluation...
:: "%VENV_PY%" "%REPO_DIR%\scripts\evaluate_model.py" --model "%MODEL_PATH%" --data "%REPO_DIR%\data\missile_dataset.csv" --out "%REPO_DIR%\assets\evaluation"

:: 8) Launch Streamlit in a new window
echo Launching Streamlit app...
start "Streamlit" cmd /k "%VENV_PY% -m streamlit run \"%REPO_DIR%\app\missile_streamlit_app.py\""

echo.
echo Done. Streamlit will open in a new terminal window.
echo If browser doesn't open automatically, copy the local URL from the new window (usually http://localhost:8501).
pause
ENDLOCAL
