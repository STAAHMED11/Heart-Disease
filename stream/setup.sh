#!/bin/bash

# Heart Disease Prediction App - Quick Setup Script

echo "=================================="
echo "Heart Disease Prediction App Setup"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 is installed"
python3 --version
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created"
else
    echo "❌ Failed to create virtual environment"
    exit 1
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo ""

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Check if dataset exists
if [ -f "heart_2020_cleaned.csv" ]; then
    echo "✅ Dataset found: heart_2020_cleaned.csv"
else
    echo "⚠️  Dataset not found!"
    echo ""
    echo "Please download the dataset:"
    echo "1. Visit: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease"
    echo "2. Download 'heart_2020_cleaned.csv'"
    echo "3. Place it in this directory"
    echo ""
fi

echo ""
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "To run the application:"
echo "1. Ensure dataset is in this directory (heart_2020_cleaned.csv)"
echo "2. Run: streamlit run heart_disease_app.py"
echo ""
echo "Or use the run script: ./run.sh"
echo ""
