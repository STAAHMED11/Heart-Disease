@echo off
echo ==================================
echo Heart Disease Prediction App Setup
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version
echo.

REM Create virtual environment
echo [*] Creating virtual environment...
python -m venv venv

if %errorlevel% equ 0 (
    echo [OK] Virtual environment created
) else (
    echo [X] Failed to create virtual environment
    pause
    exit /b 1
)
echo.

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

if %errorlevel% equ 0 (
    echo [OK] Virtual environment activated
) else (
    echo [X] Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo [*] Installing dependencies...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo [OK] All dependencies installed successfully
) else (
    echo [X] Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Check if dataset exists
if exist "heart_2020_cleaned.csv" (
    echo [OK] Dataset found: heart_2020_cleaned.csv
) else (
    echo [!] Dataset not found!
    echo.
    echo Please download the dataset:
    echo 1. Visit: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    echo 2. Download 'heart_2020_cleaned.csv'
    echo 3. Place it in this directory
    echo.
)

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To run the application:
echo 1. Ensure dataset is in this directory (heart_2020_cleaned.csv)
echo 2. Run: run.bat
echo.
pause
