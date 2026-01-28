# 📊 ChurnGuard: AI Customer Churn Prediction

A Streamlit web application that predicts customer churn using machine learning models. Built with Logistic Regression and Linear Regression algorithms to analyze customer behavior and identify those at risk of leaving.

## 🚀 Features

- **Smart Prediction**: Uses trained ML models to predict customer churn probability
- **Dual Algorithms**: Compare results between Logistic Regression and Linear Probability Model
- **Interactive Dashboard**: Clean, modern interface with intuitive controls
- **Data Visualization**: Explore churn distribution and feature correlations
- **Real-time Analysis**: Instant predictions based on customer profile inputs

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Python** - Core programming language
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

## 📁 Project Structure

```
churn-prediction-app/
├── app.py                 # Main Streamlit application
├── analysis_and_train.py  # Data preprocessing and model training
├── churn.csv             # Dataset
├── model_logistic.pkl    # Trained Logistic Regression model
├── model_linear.pkl      # Trained Linear Regression model
├── scaler.pkl           # Feature scaler
├── columns.pkl          # Feature column names
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/churn-prediction-app.git
   cd churn-prediction-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (if needed)
   ```bash
   python analysis_and_train.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. Open your browser to `http://localhost:8501`

## ☁️ Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Deploy!

### Option 2: Heroku

1. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Render

1. Create `render.yaml`:
   ```yaml
   services:
     - type: web
       name: churn-app
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

## 🎯 How to Use

1. **Navigate** using the sidebar menu
2. **Analyze Data** to understand churn patterns
3. **Make Predictions** by entering customer details:
   - Age, Gender, Geography
   - Credit Score, Account Balance
   - Salary, Product Count
   - Membership Status
4. **View Results** with confidence scores and risk assessment

## 📈 Model Performance

- **Logistic Regression Accuracy**: ~81.2%
- **Linear Regression Accuracy**: Comparable performance
- **Features Used**: 11 key customer attributes
- **Training Data**: 10,000 customer records

## 🔧 Customization

Want to improve the model?
- Add more features to the dataset
- Try different algorithms (Random Forest, XGBoost)
- Tune hyperparameters
- Handle class imbalance with SMOTE

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset sourced from banking customer records
- Built with ❤️ using Streamlit and scikit-learn