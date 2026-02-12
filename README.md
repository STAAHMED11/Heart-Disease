# ❤️ Heart Disease Prediction System

A comprehensive machine learning project for predicting heart disease risk using health and lifestyle indicators. This project includes both an interactive **Streamlit web application** and detailed **Jupyter notebook analysis**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Screenshots](#screenshots)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a machine learning-based heart disease prediction system that analyzes 18 different health and lifestyle factors to assess an individual's risk of developing heart disease. The system features:

- **Interactive Web Application**: User-friendly Streamlit interface for data exploration and predictions
- **Multiple ML Models**: Comparison of 7 different machine learning algorithms
- **Comprehensive Analysis**: Detailed exploratory data analysis with interactive visualizations
- **Real-time Predictions**: Instant heart disease risk assessment with probability scores

## ✨ Features

### 🌐 Streamlit Web Application

- **Data Exploration**
  - Dataset overview with key statistics
  - Statistical summaries of numerical and categorical features
  - Missing values analysis

- **Interactive Visualizations**
  - Target distribution analysis
  - Gender, age, and race-based heart disease rates
  - BMI, sleep, and mental health distributions
  - General health and diabetic status analysis
  - Binary health factors impact visualization
  - Correlation heatmaps
  - Lifestyle factors analysis

- **Model Training & Evaluation**
  - Train multiple ML models with one click
  - Automated dataset balancing
  - Performance comparison across all metrics
  - Feature importance visualization
  - Model persistence for future predictions

- **Prediction System**
  - User-friendly input form for health data
  - Real-time risk assessment
  - Probability scores with visual gauge
  - Personalized health recommendations

### 📊 Jupyter Notebook Analysis

- Comprehensive exploratory data analysis (EDA)
- Feature engineering and encoding strategies
- Class balancing techniques
- Detailed model training and evaluation
- Confusion matrices and classification reports
- Feature importance analysis

## 📊 Dataset

**Source**: CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020

**Size**: 319,795 survey responses

**Features**: 18 health and lifestyle indicators

**Target Variable**: HeartDisease (Yes/No)

### Key Features:
- **Demographics**: Age, Sex, Race
- **Physical Health**: BMI, Physical Health Days, Difficulty Walking
- **Mental Health**: Mental Health Days, Sleep Time
- **Medical History**: Stroke, Diabetic, Asthma, Kidney Disease, Skin Cancer
- **Lifestyle**: Smoking, Alcohol Drinking, Physical Activity
- **Overall**: General Health

**Class Distribution**: 
- No Heart Disease: 91.44%
- Heart Disease: 8.56% (handled through data balancing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Option 1: Quick Setup (Recommended)

#### For Linux/Mac:
```bash
cd stream
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

#### For Windows:
```cmd
cd stream
setup.bat
run.bat
```

### Option 2: Manual Setup

1. **Clone the repository**
```bash
git clone https://github.com/STAAHMED11/Heart-Disease
cd heart_disease
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
cd stream
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run heart_disease_app.py
```

5. **Access the app**
Open your browser and navigate to: `http://localhost:8501`

## 📖 Usage

### Using the Streamlit App

1. **Explore Data**
   - Navigate to "Data Exploration" to view dataset statistics
   - Check the "Visualizations" page for interactive charts

2. **Train Models**
   - Go to "Model Training" page
   - Configure training settings (dataset balancing, test size)
   - Click "Train Models" button
   - Review performance metrics and feature importance

3. **Make Predictions**
   - Navigate to "Prediction" page
   - Fill in the health information form
   - Click "Predict" to get risk assessment
   - View probability scores and recommendations

### Using the Jupyter Notebook

1. **Open the notebook**
```bash
jupyter notebook ProjectMl.ipynb
```

2. **Run cells sequentially** to:
   - Load and explore the data
   - Perform EDA with visualizations
   - Preprocess and balance the dataset
   - Train and evaluate models
   - Compare model performance

## 🤖 Machine Learning Models

The system trains and compares 7 different algorithms:

1. **Logistic Regression** - Linear classification baseline
2. **Decision Tree** - Non-linear tree-based classifier
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble method
5. **XGBoost** - Optimized gradient boosting
6. **K-Nearest Neighbors** - Instance-based learning
7. **Gaussian Naive Bayes** - Probabilistic classifier

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Recall (Sensitivity)**: Ability to identify positive cases
- **Precision**: Accuracy of positive predictions
- **F1-Score**: Harmonic mean of precision and recall

## 📈 Results

After training on the balanced dataset, typical model performance:

| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| Random Forest | ~75% | ~75% | ~75% | ~75% |
| Gradient Boosting | ~74% | ~73% | ~75% | ~74% |
| XGBoost | ~74% | ~73% | ~74% | ~73% |
| Logistic Regression | ~73% | ~72% | ~73% | ~72% |
| KNN | ~70% | ~68% | ~72% | ~70% |
| Decision Tree | ~68% | ~67% | ~69% | ~68% |
| Naive Bayes | ~65% | ~70% | ~63% | ~66% |

*Note: Results may vary based on random seed and balancing strategy*

### Key Findings:

- **Most Important Features**: Age, General Health, Diabetic Status, BMI
- **Age is the strongest predictor** of heart disease risk
- **Lifestyle factors** (smoking, physical activity) have moderate impact
- **Multiple health conditions** significantly increase risk

## 📁 Project Structure

```
heart_disease/
│
├── stream/                          # Streamlit application folder
│   ├── heart_disease_app.py        # Main Streamlit application
│   ├── heart_2020_cleaned.csv      # Dataset
│   ├── requirements.txt            # Python dependencies
│   ├── setup.sh / setup.bat        # Setup scripts
│   ├── run.sh / run.bat            # Run scripts
│   └── README.md                   # Stream app specific readme
│
├── ProjectMl.ipynb                 # Main Jupyter notebook with full analysis
├── heart-disease-prediction.ipynb  # Alternative notebook version
├── heart_2020_cleaned.csv          # Dataset (root copy)
├── newplot.png                     # Sample visualization
└── README.md                       # This file
```

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Programming language
- **Streamlit 1.31.0** - Web application framework
- **Pandas 2.0.3** - Data manipulation
- **NumPy 1.24.3** - Numerical computing

### Machine Learning
- **scikit-learn 1.3.0** - ML models and preprocessing
- **XGBoost 2.0.3** - Gradient boosting implementation

### Visualization
- **Plotly 5.17.0** - Interactive charts
- **Matplotlib 3.7.2** - Static plotting
- **Seaborn 0.12.2** - Statistical visualizations

## 🖼️ Screenshots

### Home Page
The welcoming landing page with project overview and key statistics.

### Data Exploration
Interactive tables showing dataset statistics, numerical summaries, and categorical distributions.

### Visualizations
Comprehensive charts analyzing heart disease rates across different demographics, health indicators, and lifestyle factors.

### Model Training
One-click model training with real-time performance metrics and comparison charts.

### Prediction Interface
User-friendly form for entering health data and receiving instant risk assessments with probability scores.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Improvement
- Add more ML models (Neural Networks, SVM)
- Implement hyperparameter tuning with GridSearchCV
- Add SHAP values for model interpretability
- Create API endpoints for predictions
- Add user authentication and prediction history
- Implement A/B testing for model comparison
- Add export functionality for results

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020
- **Inspiration**: Various heart disease prediction research papers
- **Community**: Streamlit and scikit-learn communities for excellent documentation

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out through GitHub.

---

**Note**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

---

Made with ❤️ using Python, Streamlit, and Machine Learning
