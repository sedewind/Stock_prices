import requests
import pandas as pd
import logging

API_KEY = "4bc0346b-bf31-4c7f-8b30-2b71880f5019"

# Set up logging with a custom format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class PySimFin:
    def __init__(self, api_key):
        self.base_url = "https://backend.simfin.com/api/v3/"
        self.headers = {
            "Authorization": f"api-key {api_key}",
            "Accept": "application/json"
        }

    def get_share_prices(self, ticker: str, start: str, end: str):
        url = f"{self.base_url}companies/prices/compact"
        params = {
            'ticker': ticker,
            'ratios': 'false',
            'asreported': 'false',
            'start': start,
            'end': end
        }
        
        try:
            logging.info(f"Requesting share prices for {ticker} from {start} to {end}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            # Log the API response
            logging.debug(f"API response for {ticker}: {response.json()}")

            # Extract the data from the response
            data = response.json()

            # If data exists, return it as a DataFrame with original structure
            if len(data) > 0 and 'data' in data[0]:
                columns = data[0]['columns']
                rows = data[0]['data']
                df = pd.DataFrame(rows, columns=columns)
                
                # Ensure 'Close' column is present and numeric
                if 'Last Closing Price' in df.columns:
                    df.rename(columns={'Last Closing Price': 'Close'}, inplace=True)
                
                # Try to convert numeric columns to proper numeric type
                for col in df.columns:
                    if col != 'Date':
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            pass  # Keep as is if conversion fails
                
                logging.info(f"Successfully fetched share prices for {ticker}")
                return df
            else:
                logging.error(f"Unexpected structure, cannot find price data for {ticker}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching share prices for {ticker}: {e}")
            return None

def transform_api_data_for_ml(df):
    """
    Transform API response data to match the ML model's training data format:
    Date,Close,Close_t-3,Close_t-2,Close_t-1,Target,Predicted_Target
    
    Where Target is 1 if Close > Close_t-1, otherwise 0
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock price data from API
    
    Returns:
    pandas.DataFrame: Transformed DataFrame in the format required for ML model
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Handle column names based on API response format
    if 'Date' not in df.columns:
        if 0 in df.columns:
            # Assuming the first column is Date
            df.rename(columns={0: 'Date'}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
    
    if 'Close' not in df.columns:
        # Try different possible column names for Close price
        if 3 in df.columns:
            df.rename(columns={3: 'Close'}, inplace=True)
        elif 'Last Closing Price' in df.columns:
            df.rename(columns={'Last Closing Price': 'Close'}, inplace=True)
        elif 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
    
    # Make sure Close column is numeric
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    except (KeyError, ValueError):
        # If Close column doesn't exist or cannot be converted
        logging.error("Could not convert Close column to numeric")
        # Try to find another numeric column to use
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df['Close'] = df[numeric_cols[0]]
        else:
            # Last resort - use the first non-date column
            non_date_cols = [col for col in df.columns if col != 'Date']
            if non_date_cols:
                df['Close'] = pd.to_numeric(df[non_date_cols[0]], errors='coerce')
    
    # Convert Date to datetime if it's not already
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # If we don't have both required columns, return empty DataFrame with correct structure
    if 'Date' not in df.columns or 'Close' not in df.columns or df['Close'].isna().all():
        logging.error("Missing required columns (Date or Close)")
        return pd.DataFrame(columns=['Date', 'Close', 'Close_t-3', 'Close_t-2', 'Close_t-1', 'Target', 'Predicted_Target'])
    
    # Sort by date in ascending order
    df = df.sort_values('Date')
    
    # Create lag features (t-1, t-2, t-3)
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    df['Close_t-3'] = df['Close'].shift(3)
    
    # Create target column: 1 if Close > Close_t-1, otherwise 0
    df['Target'] = (df['Close'] > df['Close_t-1']).astype(int)
    
    # Add Predicted_Target column (initially empty)
    df['Predicted_Target'] = None
    
    # Drop rows with NaN values (first three rows will have NaNs due to shift operation)
    df = df.dropna(subset=['Close_t-3', 'Close_t-2', 'Close_t-1'])
    
    # Select and reorder columns to match ML model format
    result_df = df[['Date', 'Close', 'Close_t-3', 'Close_t-2', 'Close_t-1', 'Target', 'Predicted_Target']]
    
    return result_df

def simulate_trading_strategy(df, strategy="Buy & Hold", starting_cash=10000.0, diff_threshold=0.02):
    """
    Simulates different trading strategies based on ML predictions
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and ML predictions
    strategy (str): Name of the strategy to simulate ('Buy & Hold', 'Buy/Sell', 'Diff-Based Buy/Sell')
    starting_cash (float): Initial cash balance
    diff_threshold (float): The difference threshold for 'Diff-Based Buy/Sell' strategy
    
    Returns:
    tuple: (stats_dict, portfolio_df) with performance metrics and portfolio history
    """
    if strategy == "Buy & Hold":
        # Buy 1 stock when Predicted_Target = 1, otherwise hold
        return simulate_buy_and_hold(df, starting_cash)
    
    elif strategy == "Buy/Sell":
        # Buy when Predicted_Target = 1, sell when Predicted_Target = 0
        return simulate_buy_sell(df, starting_cash)
    
    elif strategy == "Diff-Based Buy/Sell":
        # Buy when difference between Close_t-1 and Predicted_Target is greater than threshold
        return simulate_diff_based_buy_sell(df, starting_cash, diff_threshold)
    
    else:
        # Default to Buy & Hold
        return simulate_buy_and_hold(df, starting_cash)

def simulate_buy_and_hold(df, starting_cash):
    """
    Simulates a Buy & Hold strategy based on ML predictions.
    Buy 1 share when Predicted_Target = 1, otherwise hold.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance
    
    Returns:
    tuple: (stats_dict, portfolio_df) with performance metrics and portfolio history
    """
    # Create a copy to avoid modifying the original dataframe
    portfolio_df = df.copy()
    
    # Initialize portfolio tracking columns
    portfolio_df['Cash'] = starting_cash
    portfolio_df['Shares'] = 0
    portfolio_df['Stock_Value'] = 0
    portfolio_df['Portfolio_Value'] = starting_cash
    portfolio_df['Trade'] = 'None'
    
    # Loop through each day to simulate trading
    for i in range(len(portfolio_df)):
        if i > 0:  # Copy previous day's portfolio values
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] = portfolio_df.loc[portfolio_df.index[i-1], 'Cash']
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] = portfolio_df.loc[portfolio_df.index[i-1], 'Shares']
        
        # Get current day's price and prediction
        current_price = portfolio_df.loc[portfolio_df.index[i], 'Close']
        prediction = portfolio_df.loc[portfolio_df.index[i], 'Predicted_Target']
        
        # Check if we should buy (prediction = 1 and have enough cash)
        if prediction == 1 and portfolio_df.loc[portfolio_df.index[i], 'Cash'] >= current_price:
            # Buy 1 share
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= current_price
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] += 1
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Buy'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        
        # Update stock value and total portfolio value
        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']
    
    # Calculate portfolio statistics
    days_until_cash_out = None
    cash_left = portfolio_df.iloc[-1]['Cash']
    
    if cash_left <= 0:
        # Find the day when cash ran out
        zero_cash_days = portfolio_df[portfolio_df['Cash'] <= 0].index
        if not zero_cash_days.empty:
            first_zero_day = zero_cash_days[0]
            days_until_cash_out = (first_zero_day - portfolio_df.index[0]).days
    
    # Portfolio performance metrics
    initial_cash = starting_cash
    final_cash = portfolio_df.iloc[-1]['Cash']
    stock_value = portfolio_df.iloc[-1]['Stock_Value']
    total_value = portfolio_df.iloc[-1]['Portfolio_Value']
    portfolio_gain = total_value - initial_cash
    portfolio_gain_pct = (portfolio_gain / initial_cash) * 100 if initial_cash > 0 else 0
    total_shares = portfolio_df.iloc[-1]['Shares']
    
    stats = {
        'initial_cash': initial_cash,
        'final_cash': final_cash,
        'stock_value': stock_value,
        'total_value': total_value,
        'portfolio_gain': portfolio_gain,
        'portfolio_gain_pct': portfolio_gain_pct,
        'total_shares': total_shares,
        'days_until_cash_out': days_until_cash_out
    }
    
    return stats, portfolio_df

def simulate_buy_sell(df, starting_cash):
    """
    Simulates a Buy/Sell strategy based on ML predictions.
    Buy 1 share when Predicted_Target = 1, Sell when Predicted_Target = 0.

    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance

    Returns:
    tuple: (stats_dict, portfolio_df) with performance metrics and portfolio history
    """
    portfolio_df = df.copy()

    # Initialize portfolio tracking columns
    portfolio_df['Cash'] = starting_cash
    portfolio_df['Shares'] = 0
    portfolio_df['Stock_Value'] = 0
    portfolio_df['Portfolio_Value'] = starting_cash
    portfolio_df['Trade'] = 'None'

    for i in range(len(portfolio_df)):
        if i > 0:  # Copy previous day's portfolio values
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] = portfolio_df.loc[portfolio_df.index[i-1], 'Cash']
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] = portfolio_df.loc[portfolio_df.index[i-1], 'Shares']

        current_price = portfolio_df.loc[portfolio_df.index[i], 'Close']
        prediction = portfolio_df.loc[portfolio_df.index[i], 'Predicted_Target']

        # Buy 1 share if prediction = 1
        if prediction == 1 and portfolio_df.loc[portfolio_df.index[i], 'Cash'] >= current_price:
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= current_price
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] += 1
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Buy'
        
        # Sell 1 share if prediction = 0
        elif prediction == 0 and portfolio_df.loc[portfolio_df.index[i], 'Shares'] > 0:
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] += current_price
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] -= 1
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Sell'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'

        # Update stock value and portfolio value
        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']

    stats = calculate_portfolio_stats(portfolio_df)
    return stats, portfolio_df

def simulate_diff_based_buy_sell(df, starting_cash, diff_threshold):
    """
    Simulates a Buy/Sell strategy based on price differences exceeding a threshold,
    independent of consistent predictions. Predictions serve as hints but do not block trades.

    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance
    diff_threshold (float): The difference threshold for triggering buy or sell actions

    Returns:
    tuple: (stats_dict, portfolio_df) with performance metrics and portfolio history
    """
    portfolio_df = df.copy()

    portfolio_df['Cash'] = starting_cash
    portfolio_df['Shares'] = 0
    portfolio_df['Stock_Value'] = 0
    portfolio_df['Portfolio_Value'] = starting_cash
    portfolio_df['Trade'] = 'None'

    for i in range(len(portfolio_df)):
        if i > 0:
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] = portfolio_df.loc[portfolio_df.index[i-1], 'Cash']
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] = portfolio_df.loc[portfolio_df.index[i-1], 'Shares']

        current_price = portfolio_df.loc[portfolio_df.index[i], 'Close']
        prev_price = portfolio_df.loc[portfolio_df.index[i], 'Close_t-1']
        prediction = portfolio_df.loc[portfolio_df.index[i], 'Predicted_Target']

        price_diff = abs(current_price - prev_price)

        if price_diff >= diff_threshold:
            if current_price > prev_price and portfolio_df.loc[portfolio_df.index[i], 'Cash'] >= current_price:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] += 1
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Buy'

            elif current_price < prev_price and portfolio_df.loc[portfolio_df.index[i], 'Shares'] > 0:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] += current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] -= 1
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Sell'
            else:
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'

        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']

    stats = calculate_portfolio_stats(portfolio_df)
    return stats, portfolio_df


def calculate_portfolio_stats(portfolio_df):
    initial_cash = portfolio_df.iloc[0]['Cash'] + portfolio_df.iloc[0]['Stock_Value']
    final_cash = portfolio_df.iloc[-1]['Cash'] + portfolio_df.iloc[-1]['Stock_Value']
    total_value = final_cash
    portfolio_gain = total_value - initial_cash
    portfolio_gain_pct = (portfolio_gain / initial_cash) * 100 if initial_cash > 0 else 0
    total_shares = portfolio_df.iloc[-1]['Shares']

    return {
        'initial_cash': initial_cash,
        'final_cash': final_cash,
        'portfolio_gain': portfolio_gain,
        'portfolio_gain_pct': portfolio_gain_pct,
        'total_shares': total_shares
    }


# Add this section to allow the file to be imported or run directly
if __name__ == "__main__":
    client = PySimFin(API_KEY)
    df = client.get_share_prices("AAPL", "2023-01-01", "2023-12-31")
    if df is not None:
        ml_df = transform_api_data_for_ml(df)
        print(ml_df.head())