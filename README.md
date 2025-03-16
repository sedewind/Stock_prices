# OptiTrade: Machine Learning-Based Stock Price Prediction

**🔗 Access our App: [https://optitrade-iepython2.streamlit.app/](https://optitrade-iepython2.streamlit.app/)**

## 🚀 Project Overview

OptiTrade is a machine learning-powered application that predicts stock price movements through an interactive Streamlit web interface. The project combines advanced data analytics with user-friendly visualization to help traders and investors make informed decisions.

The system consists of two main components:

1. **Data Analytics Module** - Processes historical stock data, engineers features, and trains machine learning models
2. **Streamlit Web Application** - Provides interactive interface for stock analysis, predictions, and trading strategy simulation

By leveraging technical indicators and machine learning algorithms, OptiTrade offers automated insights for multiple stock tickers.

The following parts allows users to recreate this project locally if not using our web hosted app (linked above).

## 📌 Part 1: Data Analytics Module

### 🔍 Overview

This Python script automates the end-to-end process of preparing stock market data, training predictive models, and generating reliable forecasts. It uses historical stock price data to create features based on log returns, trains logistic regression models, and outputs predictions alongside performance metrics.

### 🔄 Workflow

1. Loading and filtering US stock company and share price datasets
2. Generating machine learning datasets with lagged log return features
3. Training separate logistic regression models for each stock ticker
4. Evaluating model performance with confusion matrices
5. Saving predictions, performance metrics, and serialized models

### 📊 Supported Stock Tickers

The system currently supports predictions for:
- `AMZN` (Amazon)
- `AAPL` (Apple)
- `NFLX` (Netflix)
- `MSFT` (Microsoft)
- `GOOG` (Google)

> **Note:** The script filters tickers with at least 200 data points to ensure enough historical data for training. If a ticker has insufficient data after preprocessing, it will be skipped.

### 📁 Output Files

The analytics module automatically generates the following in the `output/` directory:

| File Type | Description | Format |
|-----------|-------------|--------|
| Predictions | Stock movement forecasts | `predictions_<TICKER>.csv` |
| Performance Metrics | Model evaluation results | Confusion matrix (PNG) |
| Model Checkpoints | Trained model files | `stock_predictor_<TICKER>.pkl` |
| Scalers | Data normalization parameters | `scaler_<TICKER>.pkl` |
| Logs | Processing records | `output_log.txt` |

### ⚙️ Technical Details

- The prediction target is a binary classification indicating whether the stock's log return will be positive (1) or negative (0)
- Features are created using lagged log returns from previous trading days
- Logistic regression models are trained separately for each stock ticker
- Performance is evaluated using confusion matrices for both training and test datasets

### 🔧 Installation & Setup

#### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2️⃣ Ensure Data Availability
Ensure the required data files are located in the project directory:
- `us-companies.csv` (semicolon-delimited)
- `us-shareprices-daily.csv` (semicolon-delimited)

#### 3️⃣ Run the Analytics Script
```bash
python "Part 1 - Data Analytics Module Script - Final copy.py"
```

#### 4️⃣ Check Results
All outputs will be stored in the `output/` directory. 
Please move these files to the pkl/ if you desire updated trained ML models via pkl files: 
scalar_{ticker}.pkl, stock_predicter_{ticker}.pkl

## 📌 Part 2: Streamlit Web Application

### 🌐 Overview

The interactive web application allows users to explore predictions, visualize stock data, and simulate different trading strategies.

### 🔹 Features

#### Home Page
- Project introduction and application overview
- Navigation guidance and quick-start instructions
- Key capabilities and usage examples

#### Meet the Team
- Team member profiles and contributions
- Professional links and contact information

#### Stocks Analysis Page
- Interactive stock ticker selection
- Customizable date range for historical analysis
- Visualization of closing price trends and log returns
- Technical indicator exploration

#### Stock Price Predictions
- ML-powered forecasts for selected tickers
- Prediction confidence metrics and visualization
- Historical performance evaluation

#### Portfolio Simulation
- Interactive trading strategy testing
- Customizable parameters:
  - Starting capital
  - Initial stock holdings
  - Strategy parameters (e.g., dip percentage)
- Performance comparison across strategies

### 📊 Trading Strategies

The application implements three distinct trading strategies:

| Strategy | Description |
|----------|-------------|
| **Buy & Hold** | Purchase stocks when an increase is predicted; hold current positions otherwise |
| **Buy & Sell** | Buy when increases are predicted; sell when decreases are predicted |
| **Buy the Dip** | Purchase when stock price drops below threshold before predicted increase |

### 🛠️ Setup & Run the Web App

#### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2️⃣ Launch the Application
```bash
streamlit run streamlit_app.py
```

#### 3️⃣ Access the Interface
Open your browser and navigate to: `http://localhost:8501`

Or visit our hosted version: [https://optitrade-iepython2.streamlit.app/](https://optitrade-iepython2.streamlit.app/)

## 📦 Dependencies

The project requires the following libraries:

```
pandas==2.2.3
numpy==2.0.2
streamlit==1.41.1
plotly==6.0.0
Pillow==11.1.0
requests==2.27.1
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
```

Install all dependencies at once using:
```bash
pip install -r requirements.txt
```

## 📋 System Requirements

- Python 3.10
- Sufficient disk space for output files
- Internet connection (for stock data retrieval in the web app)
