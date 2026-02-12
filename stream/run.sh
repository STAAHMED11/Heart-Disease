#!/bin/bash

# Heart Disease Prediction App - Run Script

echo "🚀 Starting Heart Disease Prediction App..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup.sh first: ./setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dataset exists
if [ ! -f "heart_2020_cleaned.csv" ]; then
    echo "⚠️  Warning: Dataset (heart_2020_cleaned.csv) not found!"
    echo "The app will start but won't work without the dataset."
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run Streamlit app
echo "🌐 Launching application..."
echo "The app will open in your default browser."
echo "If it doesn't open automatically, visit: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run heart_disease_app.py
