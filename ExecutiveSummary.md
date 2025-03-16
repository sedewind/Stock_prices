# OptiTrade: Machine Learning-Based Stock Price Prediction

## 📊 Group 10: Executive Summary

## 🚀 Project Overview

This project involved developing a machine learning model to predict stock price movements and deploying it as a Streamlit web application. The goal was to create an automated system where users input a stock ticker and receive predictions on future price trends based on historical data and technical indicators.

## 🔬 Methodology

The project included four main phases:

### 1. ETL (Extract, Transform, Load) Process

* Downloaded bulk historical US stock price data and company names from SimFin
* Merged data frames and cleaned the data to include essential columns (ticker symbol, closing price, dates)
* Filtered the dataset to exclude companies with insufficient historical data
* Selected Google, Netflix, Microsoft, Amazon, and Apple for analysis

### 2. Machine Learning Model

* Started with Google's ticker (GOOG), creating a structured table with lagged features (3 prior closing prices) and a target variable (1 or 0) depending on whether price increased or decreased today vs yesterday
* After backtesting, switched to using log returns to improve model prediction by removing prediction clusters (large groups of increase or decrease signals) and split data for training and testing
* Trained unique logistic regression models for each company and validated the models. Generated scalers and pickle files for deployment
* Automated the process of ETL, creation of training tables, ML training, and model extraction for all 5 tickers with a python script

### 3. Streamlit Web Application Deployment

* The application fetches data via the SimFin API, processes it for both the ML model (lagged log returns df) and UI display (closing prices, log returns, portfolio simulation)
* Pickle files are used for model scaling and prediction
* The app consists of three page types: Home (application overview), Meet the Team (team info, photos, and professional links), and Stocks (user portfolio simulation with ML model backtesting)
* After selecting a stock ticker, the application prompts users to select date ranges, trading strategies, starting cash, stock quantities, and dip% via sidebar controls

### 4. Trading Strategies

* **Buy & Hold**: Buy if the model predicts an increase and cash is available; otherwise, hold
* **Buy & Sell**: Buy if an increase is predicted and cash is available, sell if a decrease is predicted and stocks have already been purchased, hold otherwise
* **Buy the Dip**: Buy when the stock falls below a user-defined threshold % within the 3 days prior to a predicted price increase

## ⚠️ Challenges

* **Model Selection**: Balancing complexity and performance for accurate predictions
* **Parameter Tuning**: Adjusted the decision threshold to reduce misclassification and reduce prediction clusters (very long streaks of 1's or 0's)
* **Feature Engineering**: Incorporated log returns to stabilize predictions and further reduce prediction clusters
* **Managing Large Datasets**: Automated training and saving models to streamline processing ML models
* **Streamlit Enhancements**: Being able to go from ideation to the final product involved many steps, each adding increasing complexity. We integrated HTML/CSS for better visuals on the logos and team images to meet desired visual UI. Features such as implementing multiple trading strategies each with their unique recommendations, and changing the quantities of stocks purchased and sold both added to the complexity of the project
* **Trading Algorithm Complexity**: Developed varied strategies to compare different strategies against each other and test the models

## 📝 Conclusions

The project successfully predicted stock price movements and provided an interactive Streamlit application. Multiple trading strategies offered flexibility based on user risk tolerance. This project reinforced understanding of the ML pipeline, from data preprocessing to model deployment.

## 🔗 [Live Demo Available](https://optitrade-iepython2.streamlit.app/)
