@echo off
title AcademiQ Predict — Student Performance Prediction System

echo =========================================
echo  AcademiQ Predict
echo  Student Performance Prediction System
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Please install Python 3.9-3.12 from https://www.python.org
    pause
    exit /b 1
)

REM Install / update dependencies
echo [1/2] Checking dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check

echo [2/2] Launching application...
echo.
echo  The app will open in your browser at:
echo  http://localhost:8501
echo.
echo  Default login credentials:
echo    Admin   : admin   / admin123
echo    Student : student / student123
echo.
echo  Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false

pause
