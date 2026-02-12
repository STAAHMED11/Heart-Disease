import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.utils import resample
import xgboost

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4B4B;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin: 20px 0;
    }
    .custom-table th {
        background-color: #f0f2f6;
        padding: 10px;
        text-align: left;
        border: 1px solid #ddd;
        font-weight: bold;
    }
    .custom-table td {
        padding: 8px 10px;
        border: 1px solid #ddd;
    }
    .custom-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .custom-table tr:hover {
        background-color: #f5f5f5;
    }
    .table-container {
        max-height: 600px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data():
    """Load the heart disease dataset"""
    try:
        df = pd.read_csv('heart_2020_cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please upload 'heart_2020_cleaned.csv'")
        return None

@st.cache_data
def preprocess_data(df, balance=True):
    """Preprocess and encode the data"""
    if df is None:
        return None, None, None, None, None
    
    # Separate features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Balance dataset if requested
    if balance:
        # Combine for resampling
        df_majority = df[df.HeartDisease == 'No']
        df_minority = df[df.HeartDisease == 'Yes']
        
        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                          replace=False,
                                          n_samples=len(df_minority),
                                          random_state=42)
        
        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
        # Separate features and target again
        X = df_balanced.drop('HeartDisease', axis=1)
        y = df_balanced['HeartDisease']
    
    # Encode categorical features
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, (label_encoders, le_target, scaler, X.columns)

@st.cache_resource
def train_models(X_train, y_train):
    """Train multiple ML models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'XGBoost': xgboost.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_models = {}
    with st.spinner('Training models...'):
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models"""
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred) * 100
        recall = metrics.recall_score(y_test, y_pred) * 100
        precision = metrics.precision_score(y_test, y_pred) * 100
        f1 = metrics.f1_score(y_test, y_pred) * 100
        
        results.append({
            'Model': name,
            'Accuracy (%)': accuracy,
            'Recall (%)': recall,
            'Precision (%)': precision,
            'F1-Score (%)': f1
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score (%)', ascending=False).reset_index(drop=True)
    
    return results_df

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Visualizations", "Model Training", "Prediction", "About"]
)

# Main App
if page == "Home":
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    st.write("""
    ### Welcome to the Heart Disease Prediction Application
    
    This application uses machine learning to predict the likelihood of heart disease based on various health 
    and lifestyle factors. The system analyzes 18 different features including BMI, smoking habits, physical 
    activity, and more.
    
    #### Features:
    - 📊 **Data Exploration**: Explore the dataset and understand its characteristics
    - 📈 **Visualizations**: Interactive charts and graphs for data analysis
    - 🤖 **Model Training**: Train and compare multiple machine learning models
    - 🔮 **Prediction**: Make predictions using the trained models
    
    #### Dataset Information:
    - **Total Samples**: 319,795
    - **Features**: 18 health and lifestyle indicators
    - **Target**: Heart Disease (Yes/No)
    - **Class Distribution**: Imbalanced (91.44% No, 8.56% Yes)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📊 Data Points**\n\n319,795 samples")
    with col2:
        st.info("**🎯 Features**\n\n18 health indicators")
    with col3:
        st.info("**🤖 Models**\n\n7 ML algorithms")

elif page == "Data Exploration":
    st.markdown('<h1 class="main-header">📊 Data Exploration</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Statistical Summary", "Missing Values"])
        
        with tab1:
            st.subheader("Dataset Preview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
                st.metric("Total Columns", df.shape[1])
            with col2:
                st.metric("Heart Disease Cases", f"{(df['HeartDisease'] == 'Yes').sum():,}")
                st.metric("Healthy Cases", f"{(df['HeartDisease'] == 'No').sum():,}")
            
            # Add row selector
            num_rows = st.slider("Number of rows to display", 10, 200, 50, 10)
            st.write(f"**Showing first {num_rows} rows:**")
            
            # Use pure HTML to completely bypass Streamlit's rendering
            df_display = df.head(num_rows)
            
            html = """
            <style>
                .custom-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                    margin: 20px 0;
                }
                .custom-table th {
                    background-color: #f0f2f6;
                    padding: 10px;
                    text-align: left;
                    border: 1px solid #ddd;
                    font-weight: bold;
                }
                .custom-table td {
                    padding: 8px 10px;
                    border: 1px solid #ddd;
                }
                .custom-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .custom-table tr:hover {
                    background-color: #f5f5f5;
                }
                .table-container {
                    max-height: 600px;
                    overflow-y: auto;
                    overflow-x: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
            </style>
            <div class="table-container">
                <table class="custom-table">
                    <thead>
                        <tr>
            """
            
            # Add headers
            for col in df_display.columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"
            
            # Add rows
            for idx, row in df_display.iterrows():
                html += "<tr>"
                for val in row:
                    html += f"<td>{val}</td>"
                html += "</tr>"
            
            html += "</tbody></table></div>"
            
            st.markdown(html, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Statistical Summary")
            
            # Numerical features
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                st.write("**Numerical Features:**")
                desc_df = df[numerical_cols].describe()
                
                # Convert to HTML
                html = desc_df.to_html(classes='custom-table', float_format='%.2f')
                st.markdown(f'<div class="table-container">{html}</div>', unsafe_allow_html=True)
            
            # Categorical features
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write("**Categorical Features:**")
                selected_cat = st.selectbox("Select a categorical feature", categorical_cols)
                value_counts = df[selected_cat].value_counts()
                
                # Display as HTML table - limited to top 20
                vc_df = pd.DataFrame({
                    selected_cat: value_counts.head(20).index,
                    'Count': value_counts.head(20).values
                })
                
                html = vc_df.to_html(classes='custom-table', index=False)
                st.markdown(f'<div class="table-container">{html}</div>', unsafe_allow_html=True)
        
        with tab3:
            st.subheader("Missing Values Analysis")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                st.success("✅ No missing values found in the dataset!")
            else:
                missing_data = missing[missing > 0]
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values
                })
                
                html = missing_df.to_html(classes='custom-table', index=False)
                st.markdown(f'<div class="table-container">{html}</div>', unsafe_allow_html=True)

elif page == "Visualizations":
    st.markdown('<h1 class="main-header">📈 Data Visualizations</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Target Distribution", "Age Distribution", "BMI Analysis", "Correlation Heatmap", 
             "Health Indicators", "Lifestyle Factors"]
        )
        
        if viz_type == "Target Distribution":
            st.subheader("Heart Disease Distribution")
            
            fig = px.pie(
                df, 
                names='HeartDisease',
                title='Heart Disease Distribution',
                hole=0.4,
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart
            counts = df['HeartDisease'].value_counts()
            fig2 = px.bar(
                x=counts.index,
                y=counts.values,
                labels={'x': 'Heart Disease', 'y': 'Count'},
                title='Heart Disease Count',
                color=counts.index,
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "Age Distribution":
            st.subheader("Age Category Analysis")
            
            age_heart = df.groupby(['AgeCategory', 'HeartDisease']).size().reset_index(name='Count')
            fig = px.bar(
                age_heart,
                x='AgeCategory',
                y='Count',
                color='HeartDisease',
                title='Heart Disease by Age Category',
                barmode='group',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "BMI Analysis":
            st.subheader("BMI Distribution")
            
            fig = px.box(
                df,
                x='HeartDisease',
                y='BMI',
                title='BMI Distribution by Heart Disease Status',
                color='HeartDisease',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.histogram(
                df,
                x='BMI',
                color='HeartDisease',
                title='BMI Histogram',
                nbins=50,
                marginal='box',
                color_discrete_sequence=['#00CC96', '#EF553B']
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "Correlation Heatmap":
            st.subheader("Feature Correlation Heatmap")
            
            # Encode categorical variables for correlation
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
            
            corr_matrix = df_encoded.corr()
            
            fig = px.imshow(
                corr_matrix,
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(height=800, width=800)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Health Indicators":
            st.subheader("Health Indicators Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stroke
                stroke_data = df.groupby(['Stroke', 'HeartDisease']).size().reset_index(name='Count')
                fig = px.bar(
                    stroke_data,
                    x='Stroke',
                    y='Count',
                    color='HeartDisease',
                    title='Heart Disease by Stroke History',
                    barmode='group',
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Diabetes
                diabetic_data = df.groupby(['Diabetic', 'HeartDisease']).size().reset_index(name='Count')
                fig = px.bar(
                    diabetic_data,
                    x='Diabetic',
                    y='Count',
                    color='HeartDisease',
                    title='Heart Disease by Diabetic Status',
                    barmode='group',
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Lifestyle Factors":
            st.subheader("Lifestyle Factors Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Smoking
                smoking_data = df.groupby(['Smoking', 'HeartDisease']).size().reset_index(name='Count')
                fig = px.bar(
                    smoking_data,
                    x='Smoking',
                    y='Count',
                    color='HeartDisease',
                    title='Heart Disease by Smoking Status',
                    barmode='group',
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Physical Activity
                activity_data = df.groupby(['PhysicalActivity', 'HeartDisease']).size().reset_index(name='Count')
                fig = px.bar(
                    activity_data,
                    x='PhysicalActivity',
                    y='Count',
                    color='HeartDisease',
                    title='Heart Disease by Physical Activity',
                    barmode='group',
                    color_discrete_sequence=['#00CC96', '#EF553B']
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Model Training":
    st.markdown('<h1 class="main-header">🤖 Model Training & Evaluation</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is not None:
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            balance_data = st.checkbox("Balance Dataset (Recommended)", value=True)
        with col2:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Preprocessing data..."):
                X_train, X_test, y_train, y_test, encoders = preprocess_data(df, balance=balance_data)
            
            if X_train is not None:
                # Train models
                models = train_models(X_train, y_train)
                
                # Evaluate models
                st.subheader("Model Performance")
                results_df = evaluate_models(models, X_test, y_test)
                
                # Display results as HTML to avoid all rendering errors
                st.write("**Model Performance Comparison:**")
                
                # Create styled HTML table manually
                html = '<div class="table-container"><table class="custom-table"><thead><tr>'
                for col in results_df.columns:
                    html += f'<th>{col}</th>'
                html += '</tr></thead><tbody>'
                
                for idx, row in results_df.iterrows():
                    html += '<tr>'
                    for i, val in enumerate(row):
                        if i == 0:  # Model name
                            html += f'<td><strong>{val}</strong></td>'
                        else:  # Numeric values
                            html += f'<td>{val:.2f}</td>'
                    html += '</tr>'
                
                html += '</tbody></table></div>'
                st.markdown(html, unsafe_allow_html=True)
                
                # Best model
                best_model = results_df.iloc[0]
                st.success(f"🏆 Best Model: **{best_model['Model']}** with F1-Score of **{best_model['F1-Score (%)']:.2f}%**")
                
                # Visualize results
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        results_df,
                        x='Model',
                        y='F1-Score (%)',
                        title='F1-Score Comparison',
                        color='F1-Score (%)',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics comparison
                    metrics_df = results_df.melt(id_vars=['Model'], 
                                                 value_vars=['Accuracy (%)', 'Recall (%)', 'Precision (%)', 'F1-Score (%)'],
                                                 var_name='Metric', value_name='Score')
                    fig = px.line(
                        metrics_df,
                        x='Model',
                        y='Score',
                        color='Metric',
                        title='All Metrics Comparison',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (for Random Forest)
                if 'Random Forest' in models:
                    st.subheader("Feature Importance (Random Forest)")
                    
                    feature_names = encoders[3]
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': models['Random Forest'].feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Feature Importance',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Save models
                if st.button("Save Best Model"):
                    best_model_name = results_df.iloc[0]['Model']
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump((models[best_model_name], encoders), f)
                    st.success(f"✅ {best_model_name} saved successfully!")

elif page == "Prediction":
    st.markdown('<h1 class="main-header">🔮 Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    st.write("Enter patient information to predict heart disease risk:")
    
    # Load model if exists
    if os.path.exists('best_model.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model, encoders = pickle.load(f)
        
        label_encoders, le_target, scaler, feature_names = encoders
        
        # Input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
                smoking = st.selectbox("Smoking", ["No", "Yes"])
                alcohol = st.selectbox("Alcohol Drinking", ["No", "Yes"])
                stroke = st.selectbox("Stroke History", ["No", "Yes"])
                physical_health = st.slider("Physical Health (days not good in last 30)", 0, 30, 0)
                mental_health = st.slider("Mental Health (days not good in last 30)", 0, 30, 0)
            
            with col2:
                diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
                sex = st.selectbox("Sex", ["Female", "Male"])
                age = st.selectbox("Age Category", 
                                  ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                   "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"])
                race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", 
                                            "Hispanic", "Other"])
                diabetic = st.selectbox("Diabetic", ["No", "Yes", "No, borderline diabetes", "Yes (during pregnancy)"])
                physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
            
            with col3:
                gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
                sleep_time = st.slider("Sleep Time (hours)", 0, 24, 7)
                asthma = st.selectbox("Asthma", ["No", "Yes"])
                kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
                skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])
            
            submit = st.form_submit_button("Predict", type="primary")
        
        if submit:
            # Prepare input data
            input_data = pd.DataFrame({
                'BMI': [bmi],
                'Smoking': [smoking],
                'AlcoholDrinking': [alcohol],
                'Stroke': [stroke],
                'PhysicalHealth': [physical_health],
                'MentalHealth': [mental_health],
                'DiffWalking': [diff_walking],
                'Sex': [sex],
                'AgeCategory': [age],
                'Race': [race],
                'Diabetic': [diabetic],
                'PhysicalActivity': [physical_activity],
                'GenHealth': [gen_health],
                'SleepTime': [sleep_time],
                'Asthma': [asthma],
                'KidneyDisease': [kidney_disease],
                'SkinCancer': [skin_cancer]
            })
            
            # Encode categorical features
            for column in input_data.columns:
                if column in label_encoders:
                    try:
                        input_data[column] = label_encoders[column].transform(input_data[column])
                    except:
                        # Handle unseen labels
                        input_data[column] = 0
            
            # Scale features
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ High Risk")
                    st.metric("Heart Disease Risk", "YES")
                else:
                    st.success("✅ Low Risk")
                    st.metric("Heart Disease Risk", "NO")
            
            with col2:
                st.metric("Risk Probability", f"{prediction_proba[1]*100:.2f}%")
            
            with col3:
                st.metric("Healthy Probability", f"{prediction_proba[0]*100:.2f}%")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_proba[1]*100,
                title={'text': "Heart Disease Risk (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if prediction == 1:
                st.warning("""
                ### ⚠️ Recommendations:
                - Consult with a healthcare professional immediately
                - Consider lifestyle modifications (diet, exercise)
                - Monitor your health indicators regularly
                - Follow up with regular check-ups
                """)
            else:
                st.info("""
                ### ✅ Recommendations:
                - Maintain a healthy lifestyle
                - Continue regular physical activity
                - Monitor your health periodically
                - Keep up with preventive care
                """)
    else:
        st.warning("⚠️ No trained model found. Please go to the Model Training page first.")

elif page == "About":
    st.markdown('<h1 class="main-header">ℹ️ About This Application</h1>', unsafe_allow_html=True)
    
    st.write("""
    ### Heart Disease Prediction System
    
    This application uses machine learning to predict the likelihood of heart disease based on various 
    health and lifestyle factors.
    
    #### Dataset
    - **Source**: CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020
    - **Size**: 319,795 survey responses
    - **Features**: 18 health and lifestyle indicators
    
    #### Machine Learning Models
    The application trains and compares 7 different ML algorithms:
    1. Logistic Regression
    2. Random Forest Classifier
    3. Gradient Boosting Classifier
    4. Decision Tree Classifier
    5. Naive Bayes
    6. K-Nearest Neighbors
    7. XGBoost
    
    #### Features Used for Prediction
    - **Demographic**: Age, Sex, Race
    - **Physical Health**: BMI, Physical Health Days, Sleep Time
    - **Medical History**: Stroke, Diabetic, Asthma, Kidney Disease, Skin Cancer
    - **Lifestyle**: Smoking, Alcohol Drinking, Physical Activity
    - **Symptoms**: Difficulty Walking, Mental Health Days
    - **General**: General Health perception
    
    #### Performance Metrics
    - **Accuracy**: Overall correctness of predictions
    - **Recall**: Ability to identify actual heart disease cases
    - **Precision**: Accuracy of positive predictions
    - **F1-Score**: Harmonic mean of precision and recall
    
    #### Important Notes
    - This tool is for educational purposes only
    - NOT a substitute for professional medical advice
    - Always consult healthcare professionals for medical decisions
    - Predictions are based on statistical patterns, not individual diagnosis
    
    #### Data Balancing
    The original dataset is highly imbalanced (91.44% No, 8.56% Yes). We use downsampling 
    of the majority class to improve model performance on minority class detection.
    
    ---
    
    **Developed with**: Streamlit, Scikit-learn, XGBoost, Plotly
    
    **Version**: 1.0
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Heart Disease Prediction System**

Version 1.0

Built with Streamlit
""")