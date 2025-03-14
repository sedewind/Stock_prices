README.txt
===========

Stock Price Prediction Automation Script
----------------------------------------

Description:
-------------
This Python script automates the process of preparing machine learning datasets, training models, generating predictions, and saving results for multiple stock tickers. It uses historical stock price data to create features based on log returns, trains logistic regression models, and outputs predictions alongside performance evaluation metrics.

The workflow includes:
- Loading and filtering US stock company and share price datasets
- Generating machine learning datasets with lagged log return features
- Training separate logistic regression models for each stock ticker
- Saving predictions, confusion matrices, and serialized models for each ticker
- Automating the entire ETL, model training, and prediction deployment process

Processed stock tickers in this version:  
`AMZN`, `AAPL`, `NFLX`, `MSFT`, `GOOG`

Output:
--------
The script automatically creates an `output` directory containing:
- CSV files with predictions for each ticker (e.g., `predictions_AMZN.csv`)
- Confusion matrix heatmaps for train and test datasets (PNG images)
- Pickle files for each trained model and scaler (e.g., `stock_predictor_AMZN.pkl`, `scaler_AMZN.pkl`)
- A log file (`output_log.txt`) capturing the console output and status messages

Requirements:
--------------
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Before running the script, ensure you have the following CSV files in the working directory:
- `us-companies.csv`  
- `us-shareprices-daily.csv`  

Both files should be delimited with semicolons (`;`).

How to Run:
------------
1. Install the required packages (if not already installed):
2. Ensure `us-companies.csv` and `us-shareprices-daily.csv` are available in the same directory as the script.
3. Run the script:
4. Check the `output` folder for results:
- Predictions CSV files
- Confusion matrix plots
- Model and scaler pickle files
- Log file with detailed run information

Notes:
-------
- The script filters tickers with at least 200 data points to ensure enough historical data for training.
- If a ticker has insufficient data after preprocessing, it will be skipped.
- The prediction target is a binary classification indicating whether the stock's log return will be positive (1) or negative (0).


