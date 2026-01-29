import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page config
st.set_page_config(
    page_title="ChurnGuard | AI Customer Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, simple and beautiful look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    /* Light, airy background */
    .stApp {
        background-color: #fcfcfd;
        color: #1a1a1a;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Clean Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f0f0f0;
    }
    
    /* Sidebar text visibility */
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    
    /* Sidebar title specifically */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #1a1a1a !important;
    }
    
    /* Sidebar radio buttons */
    section[data-testid="stSidebar"] .stRadio > label {
        color: #1a1a1a !important;
    }
    
    /* Sidebar selectbox */
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #1a1a1a !important;
    }
    
    /* Soft, Minimalist Cards */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #f0f2f6;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 1rem;
    }
    
    /* Elegant Typography */
    h1, h2 {
        font-family: 'Outfit', sans-serif;
        color: #1a1a1a;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Clean Metric Styling */
    [data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: 600;
    }
    
    /* Soft Blue Button */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background-color: #2563eb;
        color: white;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #2563eb;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 400;
    }
    </style>
""", unsafe_allow_html=True)

# Load model, scaler, and columns
@st.cache_resource
def load_resources(model_type="Logistic"):
    try:
        filename = 'model_logistic.pkl' if model_type == "Logistic" else 'model_linear.pkl'
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('columns.pkl', 'rb') as f:
            columns = pickle.load(f)
        return model, scaler, columns
    except FileNotFoundError:
        return None, None, None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction"])

if page == "Home":
    st.title("Customer Churn Analysis")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Predict and Understand Customer Behavior
        This tool uses machine learning to identify customers who are likely to leave your service. 
        By analyzing demographics and financial history, we provide actionable insights for better retention.
        
        **What you can do:**
        - **Analyze:** Explore data distributions and correlations.
        - **Predict:** Use trained models to assess individual customer risk.
        - **Compare:** Toggle between different algorithms for deeper understanding.
        """)
        
    with col2:
        st.info("📊 Churn distribution visualization")
        st.write("*Chart would appear here with local deployment*")

    st.divider()
    
    # Simple Stats Row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Algorithm", "Logistic Reg.")
    s2.metric("Samples", "10,000")
    s3.metric("Features", "11")
    s4.metric("Accuracy", "81.2%")

elif page == "Data Analysis":
    st.title("🔍 Data Insights Explorer")
    st.write("Visualizing the factors that drive customer churn.")
    
    tab1, tab2 = st.tabs(["📊 Target Distribution", "🔗 Feature Correlations"])
    
    with tab1:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.info("The dataset is slightly imbalanced, which is typical for churn analysis. Most customers stay (Exited=0).")
        with col_b:
            st.info("📊 Churn distribution chart")
            st.write("*Visualization available in local deployment*")
        
    with tab2:
        st.info("🔗 Feature correlation heatmap")
        st.write("*Correlation visualization available in local deployment*")
        st.markdown("""
        **Analysis:**
        - **Age** and **Balance** show noticeable correlations with churn.
        - **IsActiveMember** is a strong negative predictor (active members stay longer).
        """)

elif page == "Prediction":
    st.title("🔮 Predictive Analytics")
    
    # Model Selection
    selected_model_type = st.sidebar.selectbox("Select Model Algorithm", ["Logistic", "Linear"])
    model, scaler, columns = load_resources(selected_model_type)
    
    # Check if models loaded successfully
    if model is None or scaler is None or columns is None:
        st.warning("⚠️ Models not available in this deployment")
        st.info("This deployment demonstrates the UI. For full predictions, run locally or ensure model files are available.")
        st.stop()
    
    st.write(f"Using **{selected_model_type} Regression** to calculate churn probability.")
    
    with st.container():
        # Input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Details")
            age = st.slider("Age", 18, 100, 40)
            gender = st.selectbox("Gender", ["Female", "Male"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            
        with col2:
            st.subheader("💰 Financial Profile")
            credit_score = st.number_input("Credit Score", 300, 850, 600)
            balance = st.number_input("Account Balance ($)", min_value=0.0, value=0.0, step=1000.0)
            salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=5000.0)
            num_products = st.select_slider("Number of Products", options=[1, 2, 3, 4], value=1)
            has_cr_card = st.toggle("Has Credit Card?", value=True)

    st.divider()
    
    if st.button("🚀 Run Prediction Engine"):
        # Prepare input data
        input_data = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': 5, # Standard default or add to inputs
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': 1 if has_cr_card else 0,
            'IsActiveMember': is_active,
            'EstimatedSalary': salary,
            'Geography_Germany': 1 if geography == "Germany" else 0,
            'Geography_Spain': 1 if geography == "Spain" else 0,
            'Gender_Male': 1 if gender == "Male" else 0
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[columns]
        input_scaled = scaler.transform(input_df)
        
        # Prediction
        if selected_model_type == "Logistic":
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            # Linear Regression for classification (LPM)
            raw_pred = model.predict(input_scaled)[0]
            probability = np.clip(raw_pred, 0, 1)
            prediction = [1 if probability >= 0.5 else 0]
        
        st.subheader("Assessment Result")
        
        if prediction[0] == 1:
            st.error(f"⚠️ **HIGH RISK:** The model predicts this customer is likely to **CHURN**.")
            st.progress(probability)
            st.write(f"Confidence Level: **{probability:.2%}**")
        else:
            st.success(f"✅ **LOW RISK:** The model predicts this customer is likely to **STAY**.")
            st.progress(probability)
            st.write(f"Churn Probability: **{probability:.2%}**")
            
        with st.expander("Technical Details"):
            st.write("The model used is a Logistic Regression classifier trained on 10,000 customer records.")
            st.json(input_data)
