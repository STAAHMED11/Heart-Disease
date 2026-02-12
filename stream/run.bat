@echo off
echo Starting Heart Disease Prediction App...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [X] Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dataset exists
if not exist "heart_2020_cleaned.csv" (
    echo [!] Warning: Dataset (heart_2020_cleaned.csv) not found!
    echo The app will start but won't work without the dataset.
    echo.
    set /p continue="Do you want to continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Run Streamlit app
echo [*] Launching application...
echo The app will open in your default browser.
echo If it doesn't open automatically, visit: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run heart_disease_app.py
