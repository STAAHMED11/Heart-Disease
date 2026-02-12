# Heart Disease Prediction Streamlit Application

A comprehensive machine learning web application for predicting heart disease risk based on health and lifestyle factors.

## Features

- **📊 Data Exploration**: Interactive exploration of the heart disease dataset
- **📈 Visualizations**: Dynamic charts and graphs for data analysis
- **🤖 Model Training**: Train and compare 7 different ML algorithms
- **🔮 Prediction**: Real-time heart disease risk prediction
- **📱 Responsive Design**: Works on desktop, tablet, and mobile

## Dataset

- **Source**: CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020
- **Size**: 319,795 survey responses
- **Features**: 18 health and lifestyle indicators
- **Target**: Heart Disease (Yes/No)

## Machine Learning Models

The application trains and compares the following models:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Decision Tree Classifier
5. Naive Bayes Classifier
6. K-Nearest Neighbors
7. XGBoost Classifier

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
   - Download the `heart_2020_cleaned.csv` dataset
   - Place it in the same directory as `heart_disease_app.py`
   - Dataset can be found on Kaggle: [Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)

4. **Run the application**
```bash
streamlit run heart_disease_app.py
```

5. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8501`

## Usage Guide

### 1. Home Page
- Overview of the application and dataset statistics
- Quick metrics and feature summary

### 2. Data Exploration
- View the dataset preview
- Explore statistical summaries
- Check for missing values
- Understand data distribution

### 3. Visualizations
Choose from various visualization types:
- **Target Distribution**: See the balance between heart disease cases
- **Age Distribution**: Analyze heart disease by age groups
- **BMI Analysis**: Examine BMI distribution and its relationship with heart disease
- **Correlation Heatmap**: Understand feature relationships
- **Health Indicators**: Explore stroke, diabetes, and other health factors
- **Lifestyle Factors**: Analyze smoking, physical activity, and other lifestyle choices

### 4. Model Training
- Configure training parameters (data balancing, test size)
- Train multiple ML models simultaneously
- Compare model performance metrics
- View feature importance
- Save the best performing model

### 5. Prediction
- Enter patient information through an intuitive form
- Get instant heart disease risk predictions
- View probability scores
- Receive personalized health recommendations

## Application Structure

```
heart-disease-app/
│
├── heart_disease_app.py      # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── heart_2020_cleaned.csv     # Dataset (to be downloaded)
└── best_model.pkl            # Saved model (generated after training)
```

## Features Breakdown

### Input Features for Prediction

**Demographic Information:**
- Age Category (18-24 to 80+)
- Sex (Male/Female)
- Race

**Physical Health:**
- BMI (Body Mass Index)
- Physical Health (days not good in last 30)
- Sleep Time (hours)

**Medical History:**
- Stroke History
- Diabetic Status
- Asthma
- Kidney Disease
- Skin Cancer

**Lifestyle Factors:**
- Smoking Status
- Alcohol Drinking
- Physical Activity
- Difficulty Walking

**Mental & General Health:**
- Mental Health (days not good in last 30)
- General Health Perception

## Performance Metrics

The application evaluates models using:
- **Accuracy**: Overall prediction correctness
- **Recall**: Ability to identify actual heart disease cases
- **Precision**: Accuracy of positive predictions
- **F1-Score**: Harmonic mean of precision and recall

## Important Notes

⚠️ **Medical Disclaimer**: 
- This application is for **educational and informational purposes only**
- It is **NOT a substitute for professional medical advice, diagnosis, or treatment**
- Always seek the advice of qualified health providers with questions about medical conditions
- Never disregard professional medical advice based on predictions from this tool

## Technical Details

### Data Preprocessing
- Label encoding for categorical features
- Standard scaling for numerical features
- Class balancing using downsampling (optional)
- Train-test split (80-20 default)

### Model Evaluation
- Cross-validation ready
- Multiple performance metrics
- Confusion matrix support
- Feature importance analysis (for tree-based models)

### Deployment
- Built with Streamlit for easy deployment
- Can be deployed to Streamlit Cloud, Heroku, or AWS
- Supports local and cloud hosting

## Troubleshooting

### Common Issues

**1. Dataset not found error**
- Ensure `heart_2020_cleaned.csv` is in the same directory as the app
- Check file name spelling

**2. Model not found for prediction**
- Train a model first using the "Model Training" page
- The model will be saved as `best_model.pkl`

**3. Package installation errors**
- Update pip: `pip install --upgrade pip`
- Install packages one by one if bulk installation fails
- Use virtual environment to avoid conflicts

**4. Port already in use**
- Stop other Streamlit instances
- Use a different port: `streamlit run heart_disease_app.py --server.port 8502`

## Future Enhancements

- [ ] Hyperparameter tuning interface
- [ ] Model explainability (SHAP values)
- [ ] Batch prediction from CSV upload
- [ ] Export prediction reports
- [ ] User authentication
- [ ] Prediction history tracking
- [ ] API endpoint for predictions
- [ ] Mobile app version

## Dependencies

- streamlit: Web application framework
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- xgboost: Gradient boosting
- plotly: Interactive visualizations
- matplotlib: Static visualizations
- seaborn: Statistical visualizations

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- CDC's BRFSS for the dataset
- Streamlit for the amazing web framework
- Scikit-learn and XGBoost teams for ML libraries

## Contact

For questions or feedback, please open an issue in the repository.

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Built with**: ❤️ and Python
