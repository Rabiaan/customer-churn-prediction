import streamlit as st
import pandas as pd
import numpy as np

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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Prediction Demo"])

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
        st.info("📊 Churn distribution visualization would appear here in full version")
    
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
            st.info("📊 Chart visualization would appear here in full version")
        
    with tab2:
        st.info("🔗 Correlation heatmap would appear here in full version")
        st.markdown("""
        **Analysis:**
        - **Age** and **Balance** show noticeable correlations with churn.
        - **IsActiveMember** is a strong negative predictor (active members stay longer).
        """)

elif page == "Prediction Demo":
    st.title("🔮 Predictive Analytics Demo")
    
    # Model Selection
    selected_model_type = st.sidebar.selectbox("Select Model Algorithm", ["Logistic", "Linear"])
    
    st.info("⚠️ This is a demonstration version. Model files are not included in this deployment.")
    st.info("💡 For full functionality with predictions, please run this app locally with trained models.")
    
    st.write(f"This demo shows the interface for **{selected_model_type} Regression** based predictions.")
    
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
        st.subheader("Demo Result")
        st.warning("⚠️ Prediction disabled in demo mode")
        st.info("In the full version, this would show churn probability based on your inputs.")
        st.info("To use the full prediction feature, run this app locally with the trained models.")

st.sidebar.divider()
st.sidebar.info("📋 This is a lightweight demo version optimized for cloud deployment.")