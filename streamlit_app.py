import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta
from mainproject import PySimFin, API_KEY, transform_api_data_for_ml, simulate_trading_strategy

# Set page config
st.set_page_config(
    page_title="Stock Price Viewer",
    page_icon="📈",
    layout="wide"
)

# URLs for company logos
company_logos = {
    "AAPL": "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
    "AMZN": "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
    "GOOG": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Google_logo_%282013-2015%29.svg/2560px-Google_logo_%282013-2015%29.svg.png?20140518135123",  # Correct URL for Google
    "MSFT": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png?20210729021049",  # Correct URL for Microsoft
    "NFLX": "https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg"
}

# Sidebar with page navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "AAPL", "AMZN", "GOOG", "MSFT", "NFLX"])

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('stock_predictor.pkl', 'rb') as f:
    stock_predictor = pickle.load(f)

# Function to fetch stock data
def fetch_stock_data(ticker, start_date_str, end_date_str):
    client = PySimFin(API_KEY)
    df = client.get_share_prices(ticker, start_date_str, end_date_str)
    return df

if page == "Home":
    st.title("Welcome to the Stock Price Viewer App!")

    # Add custom CSS to align images by their bottom edge within the columns
    st.markdown(
        """
        <style>
        .image-container {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            height: 120px;  /* Adjust this value to fit your image size */
        }
        .company-logo {
            max-width: 100%;
            max-height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Display logos in a row under the title with consistent width and bottom alignment
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.markdown(f'''
        <div class="image-container">
            <img src="{company_logos["AAPL"]}" width="70" class="company-logo">
        </div>
        <p style="text-align: center;">Apple</p>
    ''', unsafe_allow_html=True)
    
    col2.markdown(f'''
        <div class="image-container">
            <img src="{company_logos["AMZN"]}" width="150" class="company-logo">
        </div>
        <p style="text-align: center;">Amazon</p>
    ''', unsafe_allow_html=True)
    
    col3.markdown(f'''
        <div class="image-container">
            <img src="{company_logos["GOOG"]}" width="150" class="company-logo">
        </div>
        <p style="text-align: center;">Google</p>
    ''', unsafe_allow_html=True)
    
    col4.markdown(f'''
        <div class="image-container">
            <img src="{company_logos["MSFT"]}" width="70" class="company-logo">
        </div>
        <p style="text-align: center;">Microsoft</p>
    ''', unsafe_allow_html=True)
    
    col5.markdown(f'''
        <div class="image-container">
            <img src="{company_logos["NFLX"]}" width="150" class="company-logo">
        </div>
        <p style="text-align: center;">Netflix</p>
    ''', unsafe_allow_html=True)


    st.markdown("""
    This app allows you to view historical stock prices using the SimFin API and test trading strategies. 
    You can select a stock ticker from the sidebar and specify a date range to 
    fetch the stock data. The app will display the closing price of the selected stock
    over the specified date range in an interactive chart.
    
    ### Features:
    - **View stock data for different companies**: AAPL, AMZN, GOOGL, MSFT, NFLX.
    - **Interactive charts**: See how stock prices fluctuate over time.
    - **Trading Strategies**: Simulate different trading strategies with your initial investment.
    - **Portfolio Performance**: Track your portfolio value over time.
    - **ML Predictions**: Use machine learning to guide trading decisions.
    - **Raw data**: View the raw data in a convenient table.
    """)


# Stock Pages
else:
    # Create sidebar for inputs
    st.sidebar.header("Settings")

    # Ticker selection is already handled by page selection
    default_tickers = {
        "AAPL": "Apple Inc.",
        "AMZN": "Amazon.com, Inc.",
        "GOOG": "Alphabet Inc.",
        "MSFT": "Microsoft Corporation",
        "NFLX": "Netflix, Inc."
    }
    
    ticker_name = default_tickers[page]
    
    # Add custom CSS to align the logo to the right of the title
    st.markdown(
        f"""
        <style>
        .title-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .title-container h1 {{
            margin: 0;
        }}
        .company-logo {{
            max-height: 50px;  /* Adjust the height of the logo */
            width: auto;
        }}
        </style>
        <div class="title-container">
            <h1>{ticker_name} Stock Price Viewer</h1>
            <img src="{company_logos[page]}" class="company-logo" />
        </div>
        """, unsafe_allow_html=True)

    # Date range selection
    today = datetime.now()
    default_start = today - timedelta(days=365)  # Default to 1 year of data
    default_end = today

    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=default_end)

    # Add Trading Strategy section
    st.sidebar.header("Trading Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Buy & Hold", "Buy/Sell", "Diff-Based Buy/Sell"]
    )
    
    starting_cash = st.sidebar.number_input(
        "Starting Cash ($)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=100.0
    )
    
    # Add parameter for Diff-Based strategy
    diff_threshold = 0.02  # Default value
    if strategy == "Diff-Based Buy/Sell":
        diff_threshold = st.sidebar.slider(
            "Price Difference Threshold (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.1
        )

    # Convert dates to string format required by the API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Button to trigger data fetch
    if st.sidebar.button("Fetch Stock Data"):
        # Show spinner while loading
        with st.spinner(f"Fetching data for {page}..."):
            df = fetch_stock_data(page, start_date_str, end_date_str)

            if df is not None and not df.empty:
                # Transform the data to match ML model format
                ml_df = transform_api_data_for_ml(df)
                
                # Apply the scaler and the model to make predictions
                scaled_features = scaler.transform(ml_df[['Close_t-3', 'Close_t-2', 'Close_t-1']])
                ml_df['Predicted_Target'] = stock_predictor.predict(scaled_features)

                # Display success message
                st.success(f"Successfully fetched, transformed, and predicted data for {page}")
                
                # Run the trading strategy simulation using the imported function from mainproject
                stats, portfolio_df = simulate_trading_strategy(
                    ml_df, 
                    strategy=strategy, 
                    starting_cash=starting_cash, 
                    diff_threshold=diff_threshold
                )
                
                # Display portfolio metrics
                st.header("Portfolio Performance")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("Starting Cash", f"${stats['initial_cash']:.2f}")
                col2.metric("Ending Cash", f"${stats['final_cash']:.2f}")
                if 'stock_value' in stats:
                    col3.metric("Stock Value", f"${stats['stock_value']:.2f}")
                    col4.metric("Total Portfolio Value", f"${stats['total_value']:.2f}")
                else:
                    # For the strategies that return a different stats structure
                    stock_value = portfolio_df.iloc[-1]['Stock_Value']
                    total_value = portfolio_df.iloc[-1]['Portfolio_Value']
                    col3.metric("Stock Value", f"${stock_value:.2f}")
                    col4.metric("Total Portfolio Value", f"${total_value:.2f}")
                
                # Format gain with color based on positive/negative
                gain_delta = f"{stats['portfolio_gain']:.2f} ({stats['portfolio_gain_pct']:.2f}%)"
                col5.metric("Portfolio Gain/Loss", f"${stats['portfolio_gain']:.2f}", delta=gain_delta)
                
                # Additional portfolio infoa
                st.write(f"Total shares currently held: {stats['total_shares']}")
                
                if 'days_until_cash_out' in stats and stats['days_until_cash_out'] is not None:
                    st.write(f"Cash depleted after {stats['days_until_cash_out']} days")
                else:
                    if 'final_cash' in stats:
                        st.write(f"Cash remaining at end of period: ${stats['final_cash']:.2f}")
                    else:
                        st.write(f"Cash remaining at end of period: ${portfolio_df.iloc[-1]['Cash']:.2f}")
                
                # Get the latest prediction and provide a recommendation
                if not ml_df.empty:
                    latest_data = ml_df.iloc[-1]
                    latest_prediction = latest_data['Predicted_Target']
                    latest_price = latest_data['Close']
                    latest_cash = portfolio_df.iloc[-1]['Cash']
                    latest_shares = portfolio_df.iloc[-1]['Shares']
                    latest_date = latest_data['Date'].strftime('%Y-%m-%d') if isinstance(latest_data['Date'], datetime) else latest_data['Date']
                    
                    # Create recommendation header
                    st.header("Trading Recommendation")
                    
                    # Determine action based on prediction and strategy
                    if strategy == "Buy & Hold":
                        if latest_prediction == 1 and latest_cash >= latest_price:
                            action = "BUY"
                            shares_to_trade = 1  # Current strategy buys 1 share at a time
                            max_shares = int(latest_cash / latest_price)
                            
                            recommendation = f"Based on our prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share."
                            
                            # Additional context
                            if max_shares > 1:
                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                            
                            # Use success color for buy recommendation
                            st.success(recommendation)
                        else:
                            if latest_prediction == 0:
                                action = "HOLD"
                                reason = "our model predicts the stock price may decrease on the next trading day"
                            else:
                                action = "HOLD"
                                reason = f"insufficient cash (${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                            
                            recommendation = f"Based on our analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                            
                            # Use info color for hold recommendation
                            st.info(recommendation)
                    
                    elif strategy == "Buy/Sell":
                        if latest_prediction == 1 and latest_cash >= latest_price:
                            action = "BUY"
                            shares_to_trade = 1
                            max_shares = int(latest_cash / latest_price)
                            
                            recommendation = f"Based on our Buy/Sell strategy prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share."
                            
                            if max_shares > 1:
                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                            
                            st.success(recommendation)
                        elif latest_prediction == 0 and latest_shares > 0:
                            action = "SELL"
                            shares_to_trade = 1  # Sell 1 share at a time
                            
                            recommendation = f"Based on our Buy/Sell strategy prediction, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share. You currently have {latest_shares} shares."
                            
                            st.warning(recommendation)
                        else:
                            action = "HOLD"
                            if latest_prediction == 1:
                                reason = f"insufficient cash (${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                            else:
                                reason = f"you don't have any shares to sell (current shares: {latest_shares})"
                            
                            recommendation = f"Based on our Buy/Sell strategy analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                            
                            st.info(recommendation)
                    
                    elif strategy == "Diff-Based Buy/Sell":
                        # Get previous day's price
                        prev_price = ml_df.iloc[-2]['Close'] if len(ml_df) > 1 else latest_price
                        price_diff = abs(latest_price - prev_price)
                        price_diff_pct = (price_diff / prev_price) * 100
                        
                        # Threshold comparison
                        threshold_pct = diff_threshold
                        
                        if price_diff >= diff_threshold and latest_prediction == 1 and latest_cash >= latest_price:
                            action = "BUY"
                            shares_to_trade = 1
                            max_shares = int(latest_cash / latest_price)
                            
                            recommendation = f"Based on our Diff-Based strategy, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page}. The price changed by ${price_diff:.2f} (±{price_diff_pct:.2f}%), which exceeds our threshold of {threshold_pct:.1f}%."
                            
                            if max_shares > 1:
                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                            
                            st.success(recommendation)
                        elif price_diff >= diff_threshold and latest_prediction == 0 and latest_shares > 0:
                            action = "SELL"
                            shares_to_trade = 1
                            
                            recommendation = f"Based on our Diff-Based strategy, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page}. The price changed by ${price_diff:.2f} (±{price_diff_pct:.2f}%), which exceeds our threshold of {threshold_pct:.1f}%."
                            
                            st.warning(recommendation)
                        else:
                            action = "HOLD"
                            if price_diff < diff_threshold:
                                reason = f"the price change of ${price_diff:.2f} (±{price_diff_pct:.2f}%) is below our threshold of {threshold_pct:.1f}%"
                            elif latest_prediction == 1 and latest_cash < latest_price:
                                reason = f"insufficient cash (${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                            elif latest_prediction == 0 and latest_shares <= 0:
                                reason = f"you don't have any shares to sell (current shares: {latest_shares})"
                            else:
                                reason = "current market conditions don't meet our trading criteria"
                            
                            recommendation = f"Based on our Diff-Based strategy analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                            
                            st.info(recommendation)
                    
                    # Show prediction details
                    st.write(f"**Prediction details**: For data as of {latest_date}, ML prediction score: {latest_prediction}")
                    
                    # Disclaimer
                    st.caption("This recommendation is based on our machine learning model and is for informational purposes only. Past performance does not guarantee future results.")

                # Process the original dataframe for visualization
                if all(isinstance(col, int) for col in df.columns):
                    date_col = 0
                    close_col = 3
                    plot_df = pd.DataFrame({
                        'Date': df[date_col],
                        'Close': df[close_col]
                    })
                    
                    # Convert dates if they're not already datetime objects
                    if not pd.api.types.is_datetime64_any_dtype(plot_df['Date']):
                        plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                    
                    plot_df.set_index('Date', inplace=True)
                else:
                    if 'Date' in df.columns and 'Close' in df.columns:
                        plot_df = df[['Date', 'Close']].copy()
                        
                        if not pd.api.types.is_datetime64_any_dtype(plot_df['Date']):
                            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                        
                        plot_df.set_index('Date', inplace=True)
                    elif 'date' in df.columns and 'close' in df.columns:
                        plot_df = df[['date', 'close']].copy()
                        plot_df.columns = ['Date', 'Close']
                        
                        if not pd.api.types.is_datetime64_any_dtype(plot_df['Date']):
                            plot_df['Date'] = pd.to_datetime(plot_df['Date'])
                        
                        plot_df.set_index('Date', inplace=True)
                    else:
                        st.warning("Column structure not as expected. Using available data.")
                        plot_df = df

                # Create interactive chart with Plotly showing stock price and trading activity
                try:
                    # First Figure: Stock Price and Trade Markers
                    fig1 = go.Figure()

                    # Add stock price trace
                    fig1.add_trace(go.Scatter(
                        x=portfolio_df['Date'],
                        y=portfolio_df['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='rgb(49, 130, 189)', width=3)
                    ))

                    # Add markers for buy points
                    buy_points = portfolio_df[portfolio_df['Trade'] == 'Buy']
                    if not buy_points.empty:
                        fig1.add_trace(go.Scatter(
                            x=buy_points['Date'],
                            y=buy_points['Close'],
                            mode='markers',
                            name='Buy',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ))

                    # Add markers for sell points
                    sell_points = portfolio_df[portfolio_df['Trade'] == 'Sell']
                    if not sell_points.empty:
                        fig1.add_trace(go.Scatter(
                            x=sell_points['Date'],
                            y=sell_points['Close'],
                            mode='markers',
                            name='Sell',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ))

                    # Layout for Stock Price Chart
                    fig1.update_layout(
                        title=f'{page} Stock Price with Trade Markers',
                        xaxis_title='Date',
                        yaxis_title='Stock Price ($)',
                        hovermode='x unified',
                        height=500,
                        template='plotly_white'
                    )

                    # Display Stock Price Chart
                    st.plotly_chart(fig1, use_container_width=True)

                    # Second Figure: Portfolio Value
                    fig2 = go.Figure()

                    # Add portfolio value trace
                    fig2.add_trace(go.Scatter(
                        x=portfolio_df['Date'],
                        y=portfolio_df['Portfolio_Value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='rgb(204, 102, 119)', width=3)
                    ))

                    # Layout for Portfolio Value Chart
                    fig2.update_layout(
                        title=f'{page} Portfolio Value Over Time',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value ($)',
                        hovermode='x unified',
                        height=500,
                        template='plotly_white',
                        yaxis=dict(
                            tickformat=',',  # Ensures full number format (e.g., 10356 instead of 10.356k)
                        )
                    )

                    # Display Portfolio Value Chart
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
                
                # Display trading activity in expander
                with st.expander("View Trading Activity"):
                    st.dataframe(portfolio_df[['Date', 'Close', 'Predicted_Target', 'Trade', 'Cash', 'Shares', 'Stock_Value', 'Portfolio_Value']])

                # Display strategy details based on the selected strategy
                with st.expander(f"{strategy} Strategy Explanation"):
                    if strategy == "Buy & Hold":
                        st.markdown("""
                        **Buy & Hold Strategy**
                        
                        This strategy uses machine learning predictions to make buying decisions:
                        
                        - Buy 1 share when the model predicts the stock price will increase (Predicted_Target = 1)
                        - Hold your current position when the model predicts a price decrease (Predicted_Target = 0)
                        - Never sell shares once purchased
                        
                        This approach is similar to a traditional buy and hold strategy but uses ML signals to time purchases.
                        """)
                    elif strategy == "Buy/Sell":
                        st.markdown("""
                        **Buy/Sell Strategy**
                        
                        This strategy actively trades based on machine learning predictions:
                        
                        - Buy 1 share when the model predicts the stock price will increase (Predicted_Target = 1)
                        - Sell 1 share when the model predicts the stock price will decrease (Predicted_Target = 0)
                        - Each buy/sell decision is made independently each day based on the latest prediction
                        
                        This approach attempts to capitalize on both upward and downward price movements.
                        """)
                    elif strategy == "Diff-Based Buy/Sell":
                        st.markdown(f"""
                        **Diff-Based Buy/Sell Strategy**
                        
                        This strategy combines price movement thresholds with machine learning predictions:
                        
                        - Buy 1 share when:
                          - The price change exceeds the threshold ({diff_threshold:.1f}%)
                          - The model predicts a price increase (Predicted_Target = 1)
                          - You have sufficient cash
                        
                        - Sell 1 share when:
                          - The price change exceeds the threshold ({diff_threshold:.1f}%)
                          - The model predicts a price decrease (Predicted_Target = 0)
                          - You have shares to sell
                        
                        This approach waits for significant price movements before making trades, potentially reducing the impact of market noise.
                        """)
                
                # Display ML-ready data in expander
                with st.expander("View ML-Ready Data"):
                    st.dataframe(ml_df)
                
                # Display raw data in expander
                with st.expander("View Raw Data"):
                    st.dataframe(df)
                
            else:
                st.error(f"Failed to fetch data for {page}. Please check if the ticker is valid or if the date range is valid and try again.")
    else:
        st.info("👈 Select a stock ticker, date range, and trading strategy, then click 'Fetch Stock Data' to view the results")