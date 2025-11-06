@echo off
:: ==============================================
:: Create Missile Trajectory Predictor Repo
:: Location: C:\Users\japka\OneDrive\Desktop\project\missile-trajectory-predictor
:: ==============================================

set "TARGET=C:\Users\japka\OneDrive\Desktop\project\missile-trajectory-predictor"

:: Safety check (avoid System32)
echo Current directory: %CD%
echo %CD% | findstr /I "Windows\\System32" >nul
if %ERRORLEVEL%==0 (
    echo [ERROR] Bhai tu System32 me khada hai 😭
    echo Jaa Desktop pe aur phir chalaiyo.
    pause
    exit /b 1
)

:: Create main folder
echo.
echo Creating repo folder at: %TARGET%
mkdir "%TARGET%" 2>nul
cd /d "%TARGET%"

:: Create subfolders
mkdir scripts
mkdir app
mkdir models
mkdir data
mkdir notebooks
mkdir assets

:: Create main files
echo # Missile Trajectory Predictor (Physics + Machine Learning) > README.md

:: requirements.txt
(
  echo numpy
  echo pandas
  echo scikit-learn
  echo matplotlib
  echo joblib
  echo streamlit
) > requirements.txt

:: .gitignore
(
  echo venv/
  echo __pycache__/
  echo *.pyc
  echo *.pyo
  echo *.pyd
  echo .DS_Store
  echo .ipynb_checkpoints/
  echo models/*.joblib
  echo data/*.csv
  echo *.png
) > .gitignore

:: Python scripts
echo # physics simulator + ML training > scripts\missile_ml_simple.py
echo # optional dataset generator > scripts\dataset_generator.py
echo # evaluation & plots > scripts\evaluate_model.py

:: Streamlit app
echo # Streamlit app for trajectory prediction > app\missile_streamlit_app.py

:: Notebook placeholder
echo {}> notebooks\missile_experiments.ipynb

:: Assets placeholders
echo Demo screenshot placeholder > assets\demo_screenshot.png
echo Trajectory plot placeholder > assets\trajectory_plot_example.png

:: Initialize git repo (optional)
git init >nul 2>&1
git add .
git commit -m "Initial repo structure: Missile Trajectory Predictor 🚀" >nul 2>&1

echo.
echo ✅ Repo structure created successfully at:
echo %TARGET%
echo.
echo Next Steps:
echo -----------------------------------------------
echo 1. cd /d "%TARGET%"
echo 2. python -m venv venv
echo 3. venv\Scripts\activate
echo 4. pip install -r requirements.txt
echo 5. python scripts\missile_ml_simple.py
echo 6. streamlit run app\missile_streamlit_app.py
echo -----------------------------------------------
echo.
pause
