import requests
import pandas as pd
import logging
import numpy as np

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
    Transform API response data to match the ML model's training data format using log returns:
    Date, Close, Log_Return_t-3, Log_Return_t-2, Log_Return_t-1, Target, Predicted_Target
    
    Target is 1 if Log_Return > 0, otherwise 0, Predicted_Target is initially empty.

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
            df.rename(columns={0: 'Date'}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
    
    if 'Close' not in df.columns:
        if 3 in df.columns:
            df.rename(columns={3: 'Close'}, inplace=True)
        elif 'Last Closing Price' in df.columns:
            df.rename(columns={'Last Closing Price': 'Close'}, inplace=True)
        elif 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
    
    # Ensure Close column is numeric
    try:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    except (KeyError, ValueError):
        logging.error("Could not convert Close column to numeric")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df['Close'] = df[numeric_cols[0]]
        else:
            non_date_cols = [col for col in df.columns if col != 'Date']
            if non_date_cols:
                df['Close'] = pd.to_numeric(df[non_date_cols[0]], errors='coerce')
    
    # Convert Date to datetime if it's not already
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Validate presence of essential columns
    if 'Date' not in df.columns or 'Close' not in df.columns or df['Close'].isna().all():
        logging.error("Missing required columns (Date or Close)")
        return pd.DataFrame(columns=['Date', 'Close', 'Log_Return', 'Log_Return_t-3', 'Log_Return_t-2', 'Log_Return_t-1', 'Target', 'Predicted_Target'])
    
    # Sort by date
    df = df.sort_values('Date')

    # Calculate log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Create lag features
    for lag in range(1, 4):
        df[f'Log_Return_t-{lag}'] = df['Log_Return'].shift(lag)

    # Create target variable: 1 if today's return is positive, 0 otherwise
    df['Target'] = (df['Log_Return'] > 0).astype(int)
    
    # Add Predicted_Target column (initially empty)
    df['Predicted_Target'] = None

    # Drop rows with NaN values
    df = df.dropna(subset=['Log_Return_t-3', 'Log_Return_t-2', 'Log_Return_t-1'])
    
    # Reorder columns to match ML model format
    result_df = df[['Date', 'Close', 'Log_Return', 'Log_Return_t-3', 'Log_Return_t-2', 'Log_Return_t-1', 'Target', 'Predicted_Target']]

    return result_df

def transform_api_data_for_ml_with_validation(df):
    """
    Transform API data with additional validation and error handling
    """
    try:
        # First check if we have enough data
        if df is None or len(df) < 5:
            return None, "Insufficient data for ML preparation. Need at least 5 days of price history."
        
        # Call the original transform function
        result_df = transform_api_data_for_ml(df)
        
        # Validate if the transformation was successful
        if result_df is None or result_df.empty:
            return None, "Data transformation failed. Couldn't create features."
        
        # Check if we have enough data after dropping rows with NaNs
        if len(result_df) < 3:
            return None, "After processing, not enough valid data points remain. Try extending your date range."
        
        return result_df, None
    except Exception as e:
        return None, f"Error transforming data: {str(e)}"

def simulate_trading_strategy(df, strategy="Buy & Hold", starting_cash=10000.0, dip_threshold=0.02, shares_per_trade=1):
    """
    Simulates different trading strategies based on ML predictions
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and ML predictions
    strategy (str): Name of the strategy to simulate ('Buy & Hold', 'Buy & Sell', 'Buy the Dip')
    starting_cash (float): Initial cash balance
    dip_threshold (float): The percentage drop threshold for 'Buy the Dip' strategy
    shares_per_trade (int): Number of shares to buy/sell per trade (-1 for max affordable)
    
    Returns:
    tuple: (stats_dict, portfolio_df) with performance metrics and portfolio history
    """
    if strategy == "Buy & Hold":
        return simulate_buy_and_hold(df, starting_cash, shares_per_trade)
    
    elif strategy == "Buy & Sell":
        return simulate_buy_sell(df, starting_cash, shares_per_trade)
    
    elif strategy == "Buy the Dip":
        return simulate_buy_the_dip(df, starting_cash, dip_threshold, shares_per_trade)
    
    else:
        return simulate_buy_and_hold(df, starting_cash, shares_per_trade)

def simulate_buy_and_hold(df, starting_cash, shares_per_trade=1):
    """
    Simulates a Buy & Hold strategy based on ML predictions.
    Buy shares when Predicted_Target = 1, otherwise hold.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance
    shares_per_trade (int): Number of shares to buy per trade (-1 for max affordable)
    
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
            # Determine how many shares to buy
            if shares_per_trade == -1:  # Buy maximum affordable
                max_shares = int(portfolio_df.loc[portfolio_df.index[i], 'Cash'] // current_price)
                shares_to_buy = max_shares if max_shares > 0 else 0
            else:
                # Buy specified number of shares if affordable
                cash_available = portfolio_df.loc[portfolio_df.index[i], 'Cash']
                shares_to_buy = min(shares_per_trade, int(cash_available // current_price))
            
            # Buy shares if we can afford at least one
            if shares_to_buy > 0:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= shares_to_buy * current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] += shares_to_buy
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = f'Buy {shares_to_buy}'
            else:
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        
        # Update stock value and total portfolio value
        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']
    
    # Calculate portfolio statistics
    stats = calculate_portfolio_stats(portfolio_df, starting_cash)
    return stats, portfolio_df

def simulate_buy_sell(df, starting_cash, shares_per_trade=1):
    """
    Simulates a Buy & Sell strategy based on ML predictions.
    Buy shares when Predicted_Target = 1, Sell when Predicted_Target = 0.

    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance
    shares_per_trade (int): Number of shares to buy/sell per trade (-1 for max affordable)

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

        # Buy shares if prediction = 1
        if prediction == 1 and portfolio_df.loc[portfolio_df.index[i], 'Cash'] >= current_price:
            # Determine how many shares to buy
            if shares_per_trade == -1:  # Buy maximum affordable
                max_shares = int(portfolio_df.loc[portfolio_df.index[i], 'Cash'] // current_price)
                shares_to_buy = max_shares if max_shares > 0 else 0
            else:
                # Buy specified number of shares if affordable
                cash_available = portfolio_df.loc[portfolio_df.index[i], 'Cash']
                shares_to_buy = min(shares_per_trade, int(cash_available // current_price))
            
            # Buy shares if we can afford at least one
            if shares_to_buy > 0:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= shares_to_buy * current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] += shares_to_buy
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = f'Buy {shares_to_buy}'
            else:
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        
        # Sell shares if prediction = 0
        elif prediction == 0 and portfolio_df.loc[portfolio_df.index[i], 'Shares'] > 0:
            # Determine how many shares to sell
            if shares_per_trade == -1:  # Sell all shares
                shares_to_sell = portfolio_df.loc[portfolio_df.index[i], 'Shares']
            else:
                # Sell specified number of shares if available
                shares_to_sell = min(shares_per_trade, portfolio_df.loc[portfolio_df.index[i], 'Shares'])
            
            # Sell shares if we have at least one
            if shares_to_sell > 0:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] += shares_to_sell * current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] -= shares_to_sell
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = f'Sell {shares_to_sell}'
            else:
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'

        # Update stock value and portfolio value
        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']

    stats = calculate_portfolio_stats(portfolio_df, starting_cash)
    return stats, portfolio_df

def simulate_buy_the_dip(df, starting_cash, dip_threshold, shares_per_trade=1):
    """
    Simulates a Buy the Dip strategy.
    Buy when there's a dip of specified percentage followed by a buy signal within 3 days.
    No selling allowed in this version.

    Parameters:
    df (pandas.DataFrame): DataFrame with stock data and predictions
    starting_cash (float): Initial cash balance
    dip_threshold (float): The percentage drop threshold to consider as a dip
    shares_per_trade (int): Number of shares to buy per trade (-1 for max affordable)

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
    portfolio_df['Dip_Detected'] = False
    portfolio_df['Days_Since_Dip'] = 0

    # Calculate daily price changes as percentages
    portfolio_df['Price_Change'] = portfolio_df['Log_Return']

    for i in range(len(portfolio_df)):
        if i > 0:  # Copy previous day's portfolio values
            portfolio_df.loc[portfolio_df.index[i], 'Cash'] = portfolio_df.loc[portfolio_df.index[i-1], 'Cash']
            portfolio_df.loc[portfolio_df.index[i], 'Shares'] = portfolio_df.loc[portfolio_df.index[i-1], 'Shares']
            
            # If a dip was detected, increment the counter
            if portfolio_df.loc[portfolio_df.index[i-1], 'Dip_Detected']:
                portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected'] = True
                portfolio_df.loc[portfolio_df.index[i], 'Days_Since_Dip'] = portfolio_df.loc[portfolio_df.index[i-1], 'Days_Since_Dip'] + 1
            
            # Reset counter if it's been more than 3 days since dip
            if portfolio_df.loc[portfolio_df.index[i], 'Days_Since_Dip'] > 3:
                portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected'] = False
                portfolio_df.loc[portfolio_df.index[i], 'Days_Since_Dip'] = 0

        current_price = portfolio_df.loc[portfolio_df.index[i], 'Close']
        prediction = portfolio_df.loc[portfolio_df.index[i], 'Predicted_Target']
        price_change = portfolio_df.loc[portfolio_df.index[i], 'Price_Change']

        # Check for a dip
        if price_change <= dip_threshold and not portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected']:
            portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected'] = True
            portfolio_df.loc[portfolio_df.index[i], 'Days_Since_Dip'] = 0
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Dip'
        
        # Buy if there's a positive prediction within 3 days after a dip
        elif portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected'] and prediction == 1 and portfolio_df.loc[portfolio_df.index[i], 'Cash'] >= current_price:
            # Determine how many shares to buy
            if shares_per_trade == -1:  # Buy maximum affordable
                max_shares = int(portfolio_df.loc[portfolio_df.index[i], 'Cash'] // current_price)
                shares_to_buy = max_shares if max_shares > 0 else 0
            else:
                # Buy specified number of shares if affordable
                cash_available = portfolio_df.loc[portfolio_df.index[i], 'Cash']
                shares_to_buy = min(shares_per_trade, int(cash_available // current_price))
            
            # Buy shares if we can afford at least one
            if shares_to_buy > 0:
                portfolio_df.loc[portfolio_df.index[i], 'Cash'] -= shares_to_buy * current_price
                portfolio_df.loc[portfolio_df.index[i], 'Shares'] += shares_to_buy
                portfolio_df.loc[portfolio_df.index[i], 'Dip_Detected'] = False  # Reset the dip detection
                portfolio_df.loc[portfolio_df.index[i], 'Days_Since_Dip'] = 0
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = f'Buy {shares_to_buy}'
            else:
                portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'
        else:
            portfolio_df.loc[portfolio_df.index[i], 'Trade'] = 'Hold'

        # Update stock value and portfolio value
        portfolio_df.loc[portfolio_df.index[i], 'Stock_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Shares'] * current_price
        portfolio_df.loc[portfolio_df.index[i], 'Portfolio_Value'] = portfolio_df.loc[portfolio_df.index[i], 'Cash'] + portfolio_df.loc[portfolio_df.index[i], 'Stock_Value']

    # Calculate portfolio statistics
    stats = calculate_portfolio_stats(portfolio_df, starting_cash)
    return stats, portfolio_df

def calculate_portfolio_stats(portfolio_df, original_starting_cash=None):
    """
    Calculate portfolio performance statistics
    
    Parameters:
    portfolio_df (pandas.DataFrame): DataFrame with portfolio history
    original_starting_cash (float): The original starting cash amount, if provided
    
    Returns:
    dict: Dictionary with performance metrics
    """
    # Use passed starting cash or first row cash if not provided
    initial_cash = original_starting_cash if original_starting_cash is not None else portfolio_df.iloc[0]['Cash']

    # Portfolio performance metrics
    final_cash = portfolio_df.iloc[-1]['Cash']
    stock_value = portfolio_df.iloc[-1]['Stock_Value']
    total_value = portfolio_df.iloc[-1]['Portfolio_Value']
    portfolio_gain = total_value - initial_cash
    portfolio_gain_pct = (portfolio_gain / initial_cash) * 100 if initial_cash > 0 else 0
    total_shares = portfolio_df.iloc[-1]['Shares']
    
    # Calculate days until cash out if applicable
    days_until_cash_out = None
    if final_cash <= 0:
        zero_cash_days = portfolio_df[portfolio_df['Cash'] <= 0].index
        if not zero_cash_days.empty:
            first_zero_day = zero_cash_days[0]
            days_until_cash_out = (first_zero_day - portfolio_df.index[0]).days
    
    # Calculate number of trades
    buys = (portfolio_df['Trade'] == 'Buy').sum() + (portfolio_df['Trade'] == 'Buy After Dip').sum()
    sells = (portfolio_df['Trade'] == 'Sell').sum()
    
    stats = {
        'initial_cash': initial_cash,
        'final_cash': final_cash,
        'stock_value': stock_value,
        'total_value': total_value,
        'portfolio_gain': portfolio_gain,
        'portfolio_gain_pct': portfolio_gain_pct,
        'total_shares': total_shares,
        'days_until_cash_out': days_until_cash_out,
        'total_buys': buys,
        'total_sells': sells
    }
    
    return stats