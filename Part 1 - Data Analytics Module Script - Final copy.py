import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# SETUP OUTPUT FOLDER AND LOGGING
# ======================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redirect print to a log file as well as the console
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):  # Python 3 compatibility
        pass

# Initialize logger
sys.stdout = Logger(os.path.join(OUTPUT_DIR, "output_log.txt"))

# ======================
# LOAD AND PREPARE DATA
# ======================
def load_and_prepare_data():
    us_companies = pd.read_csv("us-companies.csv", sep=";")
    us_shareprices_daily = pd.read_csv("us-shareprices-daily.csv", sep=";")

    us_companies = us_companies[["Ticker", "SimFinId", "Company Name", "Number Employees"]]

    merged_df = pd.merge(us_companies, us_shareprices_daily, on=["Ticker", "SimFinId"], how="outer")

    filtered_df = merged_df.dropna(thresh=len(merged_df.columns) - 6)

    ticker_counts = filtered_df.groupby("Ticker").size().reset_index(name="Count")
    valid_tickers = ticker_counts[ticker_counts["Count"] >= 200]["Ticker"]

    filtered_df = filtered_df[filtered_df["Ticker"].isin(valid_tickers)]

    print(f"Total valid tickers found: {len(valid_tickers)}")

    return filtered_df, valid_tickers.tolist()

# =====================
# PREPARE ML DATA - USING LOG RETURNS
# =====================
def prepare_ml_data(filtered_df, ticker):
    """
    Prepares machine learning data for the given stock ticker using log returns.
    """
    stock_df = filtered_df[filtered_df["Ticker"] == ticker][['Date', 'Close']].copy()
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df.sort_values(by='Date')

    # Calculate log returns
    stock_df['Log_Return'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))

    # Create lagged log return features
    stock_df['Log_Return_t-3'] = stock_df['Log_Return'].shift(3)
    stock_df['Log_Return_t-2'] = stock_df['Log_Return'].shift(2)
    stock_df['Log_Return_t-1'] = stock_df['Log_Return'].shift(1)

    # Define target as rise/fall based on log return
    stock_df['Target'] = (stock_df['Log_Return'] > 0).astype(int)

    # Drop NaN values from shifting and log returns
    stock_ml_df = stock_df.dropna().reset_index(drop=True)

    if len(stock_ml_df) < 50:
        print(f"Not enough data points for {ticker}. Skipping.")
        return None
    
    return stock_ml_df

# =====================
# TRAIN AND DEPLOY MODEL - LOG RETURN BASED FEATURES
# =====================
def train_and_deploy_stock_model(stock_ml_df, ticker):
    """
    Trains and deploys a stock prediction model for a given stock dataframe.
    """
    # Define consistent features (drop unnecessary columns)
    features_to_exclude = ['Date', 'Close', 'Log_Return', 'Target']
    X = stock_ml_df.drop(columns=features_to_exclude, errors='ignore')
    y = stock_ml_df['Target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    classifier = LogisticRegression(random_state=1)
    classifier.fit(X_train_scaled, y_train)

    # Predict probabilities on train and test sets
    y_train_probs = classifier.predict_proba(X_train_scaled)[:, 1]
    y_test_probs = classifier.predict_proba(X_test_scaled)[:, 1]

    # Probability stats
    print(f"\nProbability Stats for {ticker}:")
    print(f"Train Set -> Min: {np.min(y_train_probs):.4f}, Max: {np.max(y_train_probs):.4f}, Mean: {np.mean(y_train_probs):.4f}")
    print(f"Test Set  -> Min: {np.min(y_test_probs):.4f}, Max: {np.max(y_test_probs):.4f}, Mean: {np.mean(y_test_probs):.4f}")

    # Threshold based on median probability from test set
    optimal_threshold = np.median(y_test_probs)
    print(f"\nSelected threshold for {ticker} based on median: {optimal_threshold:.4f}")

    # Apply threshold to get predicted labels
    y_predict_train = (y_train_probs >= optimal_threshold).astype(int)
    y_predict_test = (y_test_probs >= optimal_threshold).astype(int)

    # ======= Save Training Confusion Matrix =======
    cm_train = confusion_matrix(y_train, y_predict_train)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{ticker} - Confusion Matrix (Train)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_train_{ticker}.png"))
    plt.close()

    print(f"{ticker} - Classification Report (Train):")
    print(classification_report(y_train, y_predict_train))

    # ======= Save Test Confusion Matrix =======
    cm_test = confusion_matrix(y_test, y_predict_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Reds")
    plt.title(f"{ticker} - Confusion Matrix (Test)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"conf_matrix_test_{ticker}.png"))
    plt.close()

    print(f"{ticker} - Classification Report (Test):")
    print(classification_report(y_test, y_predict_test))

    # ======= Save Model and Scaler =======
    model_filename = os.path.join(OUTPUT_DIR, f"stock_predictor_{ticker}.pkl")
    scaler_filename = os.path.join(OUTPUT_DIR, f"scaler_{ticker}.pkl")

    with open(model_filename, "wb") as file:
        pickle.dump(classifier, file)

    with open(scaler_filename, "wb") as file:
        pickle.dump(scaler, file)

    print(f"\nModel and scaler for {ticker} saved successfully!")

    # ======= Predict on the ENTIRE DATAFRAME =======
    X_scaled_full = scaler.transform(X)
    y_probs_full = classifier.predict_proba(X_scaled_full)[:, 1]
    y_pred_full = (y_probs_full >= optimal_threshold).astype(int)

    # Add predictions to ENTIRE dataframe
    stock_ml_df['Predicted_Target'] = y_pred_full

    print(f"\nPredictions added to the entire dataframe for {ticker} using threshold {optimal_threshold:.4f}.")

    return stock_ml_df

# =====================
# MAIN AUTOMATION LOGIC
# =====================
if __name__ == "__main__":
    
    # Load and filter the dataset
    filtered_df, valid_tickers = load_and_prepare_data()

    # Hardcoded list of tickers to process
    selected_tickers = ["AMZN", "AAPL", "NFLX", "MSFT", "GOOG"]

    for ticker in selected_tickers:
        print(f"\n\nProcessing {ticker}...")

        # Prepare dataset with log returns
        stock_ml_df = prepare_ml_data(filtered_df, ticker)

        if stock_ml_df is None:
            continue  # Skip this ticker if insufficient data

        # Train model, deploy and predict
        stock_ml_df = train_and_deploy_stock_model(stock_ml_df, ticker)

        # Save the dataframe with predictions
        output_filename = os.path.join(OUTPUT_DIR, f"predictions_{ticker}.csv")
        stock_ml_df.to_csv(output_filename, index=False)
        print(f"Predictions for {ticker} saved to {output_filename}")

    print("\n\nAll selected tickers processed successfully!")
