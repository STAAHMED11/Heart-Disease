# Quick Start Guide

## Heart Disease Prediction Streamlit App

### 🚀 Getting Started in 3 Steps

#### Step 1: Setup Environment

**For Linux/Mac:**
```bash
chmod +x setup.sh run.sh
./setup.sh
```

**For Windows:**
```bash
setup.bat
```

#### Step 2: Get the Dataset

1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
2. Download `heart_2020_cleaned.csv`
3. Place it in the app directory (same folder as `heart_disease_app.py`)

#### Step 3: Run the App

**For Linux/Mac:**
```bash
./run.sh
```

**For Windows:**
```bash
run.bat
```

**Or manually:**
```bash
streamlit run heart_disease_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

### 📱 Using the Application

#### 1️⃣ Home Page
- See overview and statistics
- Understand the dataset

#### 2️⃣ Data Exploration
- Browse the dataset
- View statistics
- Check data quality

#### 3️⃣ Visualizations
- Explore interactive charts
- Analyze patterns
- Understand correlations

#### 4️⃣ Model Training
- Configure training settings
- Train multiple models
- Compare performance
- Save best model

#### 5️⃣ Make Predictions
- Enter patient data
- Get instant predictions
- View risk assessment
- See recommendations

---

### 💡 Tips

1. **First Time Users**: Start with "Home" to understand the app
2. **Data Scientists**: Check "Model Training" to see model comparisons
3. **Healthcare Providers**: Use "Prediction" for patient risk assessment
4. **Researchers**: Explore "Visualizations" for insights

---

### ⚠️ Important Notes

- This is for **educational purposes only**
- **NOT a substitute** for professional medical advice
- Always consult healthcare professionals
- Predictions are based on patterns, not individual diagnosis

---

### 🆘 Troubleshooting

**Dataset not found?**
- Ensure `heart_2020_cleaned.csv` is in the correct directory
- Check the file name (case-sensitive on Linux/Mac)

**Model not found for predictions?**
- Go to "Model Training" page
- Click "Train Models" button
- Wait for training to complete
- Click "Save Best Model"

**Port already in use?**
```bash
streamlit run heart_disease_app.py --server.port 8502
```

**Python packages missing?**
```bash
pip install -r requirements.txt
```

---

### 📊 Example Workflow

1. **Explore**: Start with Data Exploration to understand the data
2. **Visualize**: Check Visualizations to see patterns
3. **Train**: Use Model Training to build models
4. **Predict**: Make predictions with real data
5. **Analyze**: Review results and recommendations

---

### 🎯 Key Features

✅ 7 Machine Learning Models  
✅ Interactive Visualizations  
✅ Real-time Predictions  
✅ Feature Importance Analysis  
✅ Model Performance Comparison  
✅ Risk Assessment Dashboard  
✅ Health Recommendations  

---

### 📞 Need Help?

- Check the README.md for detailed documentation
- Review the troubleshooting section
- Open an issue on GitHub

---

**Ready to get started? Run the app and explore!** 🚀
