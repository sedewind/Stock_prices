import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model and scaler
MODEL_FILENAME = "stock_predictor.pkl"
SCALER_FILENAME = "scaler.pkl"

with open(MODEL_FILENAME, "rb") as file:
    classifier = pickle.load(file)

with open(SCALER_FILENAME, "rb") as file:
    scaler = pickle.load(file)

# Load datasets
us_companies = pd.read_csv("us-companies.csv", sep=";")
us_shareprices_daily = pd.read_csv("us-shareprices-daily.csv", sep=";")

# Keep only necessary columns
us_companies = us_companies[["Ticker", "SimFinId", "Company Name", "Number Employees"]]

# Merge datasets
merged_df = pd.merge(us_companies, us_shareprices_daily, on=["Ticker", "SimFinId"], how="outer")

# Drop rows with too many missing values
filtered_df = merged_df.dropna(thresh=len(merged_df.columns) - 6)

# Keep tickers with at least 200 records
ticker_counts = filtered_df.groupby("Ticker").size().reset_index(name="Count")
valid_tickers = ticker_counts[ticker_counts["Count"] >= 200]["Ticker"]
filtered_df = filtered_df[filtered_df["Ticker"].isin(valid_tickers)]

# Function to prepare ML dataset
def prepare_ml_data(ticker):
    stock_df = filtered_df[filtered_df["Ticker"] == ticker][['Date', 'Close']].copy()
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values(by='Date')
    
    # Lagged features
    stock_df['Close_t-3'] = stock_df['Close'].shift(3)
    stock_df['Close_t-2'] = stock_df['Close'].shift(2)
    stock_df['Close_t-1'] = stock_df['Close'].shift(1)
    
    # Target variable
    stock_df['Target'] = (stock_df['Close'] > stock_df['Close'].shift(1)).astype(int)
    
    # Drop NaN values
    stock_ml_df = stock_df.dropna().reset_index(drop=True)
    return stock_ml_df

# Function to apply model
def predict_stock_movement(stock_df, ticker):
    X_stock = stock_df[['Close_t-3', 'Close_t-2', 'Close_t-1']]
    X_stock_scaled = scaler.transform(X_stock)
    stock_df['Predicted_Target'] = classifier.predict(X_stock_scaled)
    
    # Save predictions to CSV
    output_filename = f"predictions_{ticker}.csv"
    stock_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")
    
    return stock_df

if __name__ == "__main__":
    selected_tickers = ["AMZN", "AAPL", "NFLX", "MSFT", "GOOG"]
    
    for ticker in selected_tickers:
        stock_df = prepare_ml_data(ticker)
        stock_df = predict_stock_movement(stock_df, ticker)
        print(f"Predictions for {ticker}:")
        print(stock_df[['Date', 'Close', 'Predicted_Target']].tail())
