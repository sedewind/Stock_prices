import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from PIL import Image
from datetime import datetime, timedelta
from mainproject import PySimFin, API_KEY, transform_api_data_for_ml, simulate_trading_strategy, transform_api_data_for_ml_with_validation

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
page = st.sidebar.selectbox("Select a Page", ["🏠 Home", "AAPL", "AMZN", "GOOG", "MSFT", "NFLX", "👨‍💻👩‍💻 Meet the Team"])

# Function to load ticker-specific models
def load_models(ticker):
    scaler_path = f'pkl/scaler_{ticker}.pkl'
    model_path = f'pkl/stock_predictor_{ticker}.pkl'
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(model_path, 'rb') as f:
            stock_predictor = pickle.load(f)
            
        return scaler, stock_predictor
    except Exception as e:
        st.error(f"Error loading models for {ticker}: {e}")
        return None, None

# Function to fetch stock data
def fetch_stock_data(ticker, start_date_str, end_date_str):
    """Fetch stock data with error handling"""
    try:
        client = PySimFin(API_KEY)
        df = client.get_share_prices(ticker, start_date_str, end_date_str)
        
        if df is None or df.empty:
            return None, "No data returned from API."
        
        # Check if we have enough data points
        if len(df) < 5:  # We need at least 5 data points for log returns and lag features
            return None, f"Only {len(df)} data points found. We need at least 5 trading days to calculate features."
            
        return df, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

if page == "🏠 Home":
    st.title("Welcome to OptiTrade!")

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
    # Home Page

    ## Overview of the Trading System

    Welcome to **OptiTrade**, a web-based application that provides real-time stock price analysis, trading strategy simulations, and machine learning-driven predictions to assist users in making informed investment decisions.

    This application integrates **SimFin API** to fetch financial data, applies **log return transformations** for feature engineering, and leverages **machine learning models** to predict stock movements. Users can explore stock trends, simulate different trading strategies, and analyze portfolio performance using interactive charts.

    ## Core Functionalities

    - **Stock Price Viewer**: Visualize historical stock prices for **Apple (AAPL), Amazon (AMZN), Google (GOOG), Microsoft (MSFT), and Netflix (NFLX)**.
    - **Trading Strategy Simulation**: Test different trading strategies such as:
        - **Buy & Hold**: Invest based on ML predictions and hold stocks over time.
        - **Buy & Sell**: Trade stocks dynamically based on model predictions.
        - **Buy the Dip**: Identify and capitalize on short-term market dips.
    - **Machine Learning Predictions**: Utilize predictive models trained on log returns to forecast stock price movements.
    - **Portfolio Simulation**: Monitor investments, track cash flow, and evaluate gains or losses.
    - **Interactive Data Visualizations**: View stock trends and portfolio values with **Plotly-powered** charts.

    ## Development Team

    Visit our "Meet the Team" Page!

    ## System Purpose and Objectives

    The primary goal of this system is to provide traders and investors with **data-driven insights** for informed decision-making. By leveraging machine learning techniques, the platform aims to enhance trading efficiency and **minimize risk** by providing predictive analytics on stock trends.

    Whether you're a **beginner investor** looking to explore market trends or an **advanced trader** testing automated strategies, this platform offers **valuable insights** to optimize your trading approach.

    ---

    For more details, visit the stock pages by navigating through the sidebar.
                
    *Logos used in this application are sourced from Wikimedia Commons:*  
    [AAPL](https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg) | [AMZN](https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg) | [GOOG](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Google_logo_%282013-2015%29.svg/2560px-Google_logo_%282013-2015%29.svg.png?20140518135123) | [MSFT](https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png?20210729021049) | [NFLX](https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg)

    """)

elif page == "👨‍💻👩‍💻 Meet the Team":
    st.title("👨‍💻👩‍💻 Meet the Team")

    # Define the path to the team photos folder
    team_photos_path = "teamphotos/"

    # Define team members with roles, descriptions, and LinkedIn links
    team_members = [
        {
            "name": "Sebastian de Wind Pesantes", 
            "photo": "sebastian.jpeg", 
            "role": "Project Manager, Data Scientist",
            "description": "Leading the team and overseeing the ML pipeline.",
            "linkedin": "https://www.linkedin.com/in/sebastiandewind/"
        },
        {
            "name": "Sarina Ratnabhas", 
            "photo": "sarina.jpeg", 
            "role": "Frontend Developer",
            "description": "Designing and developing the UI in Streamlit.",
            "linkedin": "https://www.linkedin.com/in/sarina-ratnabhas-27a1b9292/"
        },
        {
            "name": "Shadi Alfaraj", 
            "photo": "shadi.jpeg", 
            "role": "Machine Learning Engineer, ML",
            "description": "Handling data pipelines and ML integration.",
            "linkedin": "https://www.linkedin.com/in/shadi-alfaraj-28a84b1a6/"
        },
        {
            "name": "Thomas Mann", 
            "photo": "thomas.jpeg", 
            "role": "Fullstack Developer",
            "description": "Developing the API Connection, Trading Strategies, and Streamlit functionality.",
            "linkedin": "https://www.linkedin.com/in/thomasjmann23/"
        },
        {
            "name": "Uxia Lojo Miranda", 
            "photo": "uxia.jpeg", 
            "role": "Data Analyst, ML",
            "description": "Transforming bulk data for feature selection and ML model training.",
            "linkedin": "https://www.linkedin.com/in/uxia-l-214776242/"
        },
        {
            "name": "Alex Karam", 
            "photo": "alex.jpeg",
            "role": "Data Engineer, ML",
            "description": "Data Selection and EDA analytics.",
            "linkedin": "https://www.linkedin.com/in/alexanderkaram1/"
        }
    ]

    # Display team members in a grid (3 members per row)
    num_cols = 3  # Number of columns per row
    rows = [team_members[i:i + num_cols] for i in range(0, len(team_members), num_cols)]

    for row in rows:
        cols = st.columns(len(row))
        for i, member in enumerate(row):
            photo_path = os.path.join(team_photos_path, member["photo"])

            try:
                img = Image.open(photo_path)
                cols[i].image(img, caption=member["name"], use_container_width=True)

                # Display name with LinkedIn icon next to it
                cols[i].markdown(
                    f"""
                    <div style="display: flex; align-items: center;">
                        <b>{member['name']}</b>
                        <a href="{member['linkedin']}" target="_blank">
                            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" 
                            width="16" height="16" style="margin-left: 8px;">
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Display role
                cols[i].markdown(f"**{member['role']}**")

                # Display role description
                cols[i].write(member["description"])

            except Exception as e:
                cols[i].write(f"⚠️ Error loading {member['name']}'s photo.")



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

    st.sidebar.markdown("### Date Range")
    st.sidebar.markdown("*Note: We require at least 4 days of data before your selected start date for predictions.*")

    # Get user selected dates
    selected_start_date = st.sidebar.date_input("Start Date (for viewing data)", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=default_end)

    # Adjust start date to include 4 previous days for feature calculation
    start_date = selected_start_date - timedelta(days=4)  # Get 4 extra days for preprocessing

    # Display the actual data fetch range (optional, for transparency)
    st.sidebar.markdown(f"*Data will be fetched from {start_date.strftime('%Y-%m-%d')} to prepare features*")

    # Validate date range
    min_required_days = 7  # Minimum days needed (4 for features + at least 3 for display)
    if (end_date - selected_start_date).days < 3:
        st.sidebar.error("⚠️ Please select at least 3 days between start and end dates for meaningful analysis.")

    # Add Trading Strategy section
    st.sidebar.header("Trading Strategy")
    strategy = st.sidebar.selectbox(
        "Select Strategy",
        ["Buy & Hold", "Buy & Sell", "Buy the Dip"]
    )
    
    starting_cash = st.sidebar.number_input(
        "Starting Cash ($)",
        min_value=100.0,
        max_value=1000000.0,
        value=10000.0,
        step=100.0
    )

    trade_quantity = st.sidebar.selectbox(
        "Shares Per Trade",
        ["1 share", "5 shares", "10 shares", "25 shares", "Max affordable"],
        index=0
    )
    
    # Convert selection to numeric value (needed for the simulation)
    if trade_quantity == "Max affordable":
        shares_per_trade = -1  # Special value to indicate "buy maximum"
    else:
        shares_per_trade = int(trade_quantity.split()[0])  # Extract the number
    
    # Add parameter for Dip-Based strategy
    dip_threshold = 0.02  # Default dip threshold for log returns
    if strategy == "Buy the Dip":
        dip_threshold = st.sidebar.slider(
            "Dip Threshold (Log Return)",
            min_value=-0.05,  # ~5% drop
            max_value=-0.005, # ~0.5% drop
            value=-0.02,      # Default to 2% drop in log return terms
            step=0.001
        )


    # Convert dates to string format required by the API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Fix for the "Fetch Stock Data" button section
    if st.sidebar.button("Fetch Stock Data"):
        # Show spinner while loading
        with st.spinner(f"Fetching data for {page}..."):
            # Load ticker-specific models
            scaler, stock_predictor = load_models(page)
            
            if scaler is None or stock_predictor is None:
                st.error(f"❌ Failed to load models for {page}. Please ensure the model files exist.")
            else:
                # Fetch data with error handling
                df, fetch_error = fetch_stock_data(page, start_date_str, end_date_str)
                
                if fetch_error:
                    st.error(f"❌ {fetch_error}")
                    st.info("Try adjusting your date range or check your internet connection.")
                elif df is not None:
                    # Transform the data with error handling
                    ml_df, transform_error = transform_api_data_for_ml_with_validation(df)
                    
                    if transform_error:
                        st.error(f"❌ {transform_error}")
                    else:
                        try:
                            # Apply the scaler and model to make predictions
                            features = ml_df[['Log_Return_t-3', 'Log_Return_t-2', 'Log_Return_t-1']]
                            scaled_features = scaler.transform(features)
                            ml_df['Predicted_Target'] = stock_predictor.predict(scaled_features)
                            
                            # Filter the dataframe to show only from the user-selected start date
                            display_start_date = pd.to_datetime(selected_start_date)
                            ml_df_display = ml_df[ml_df['Date'] >= display_start_date].copy()
                            
                            if len(ml_df_display) == 0:
                                st.error("❌ No data available for the selected date range after processing.")
                                st.info("The selected date range may include non-trading days or holiday periods. Try extending your date range.")
                            else:
                                # Display success message
                                st.success(f"✅ Successfully fetched and analyzed data for {page}")
                                
                                # Run the trading strategy simulation

                                stats, portfolio_df = simulate_trading_strategy(
                                    ml_df_display,  # Use the filtered dataframe for simulation
                                    strategy=strategy, 
                                    starting_cash=starting_cash, 
                                    dip_threshold=dip_threshold,
                                    shares_per_trade=shares_per_trade  # Add this line
                                )
                                
                                # Display portfolio metrics
                                st.header("Portfolio Performance")
                                col1, col2, col3, col4, col5 = st.columns(5)

                                # Use the original starting_cash parameter instead of the first row value
                                col1.metric("Starting Cash", f"${starting_cash:.2f}")
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

                                # Calculate portfolio gain based on the starting_cash parameter
                                portfolio_gain = stats['total_value'] - starting_cash if 'total_value' in stats else total_value - starting_cash
                                portfolio_gain_pct = (portfolio_gain / starting_cash) * 100

                                # Format gain with color based on positive/negative
                                gain_delta = f"{portfolio_gain:.2f} ({portfolio_gain_pct:.2f}%)"
                                col5.metric("Portfolio Gain/Loss", f"${portfolio_gain:.2f}", delta=gain_delta)

                                # Additional portfolio info
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
                                            
                                            if shares_per_trade == -1:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = max_shares
                                                recommendation = f"Based on our prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** (maximum affordable) of {page} at approximately \${latest_price:.2f} per share."
                                            else:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = min(shares_per_trade, max_shares)
                                                recommendation = f"Based on our prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share."
                                            
                                            # Additional context
                                            if max_shares > shares_to_trade and shares_per_trade != -1:
                                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                                            
                                            # Total cost
                                            total_cost = shares_to_trade * latest_price
                                            recommendation += f" Total cost: **${total_cost:.2f}**"
                                            
                                            # Use success color for buy recommendation
                                            st.success(recommendation)
                                        else:
                                            if latest_prediction == 0:
                                                action = "HOLD"
                                                reason = "our model predicts the stock price may decrease on the next trading day"
                                            else:
                                                action = "HOLD"
                                                reason = f"insufficient cash (\${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                                            
                                            recommendation = f"Based on our analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                                            
                                            # Use info color for hold recommendation
                                            st.info(recommendation)
                                    
                                    elif strategy == "Buy & Sell":
                                        if latest_prediction == 1 and latest_cash >= latest_price:
                                            action = "BUY"
                                            
                                            if shares_per_trade == -1:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = max_shares
                                                recommendation = f"Based on our Buy/Sell strategy prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** (maximum affordable) of {page} at approximately \${latest_price:.2f} per share."
                                            else:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = min(shares_per_trade, max_shares)
                                                recommendation = f"Based on our Buy/Sell strategy prediction for the next trading day, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share."
                                            
                                            # Additional context
                                            if max_shares > shares_to_trade and shares_per_trade != -1:
                                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                                            
                                            # Total cost
                                            total_cost = shares_to_trade * latest_price
                                            recommendation += f" Total cost: **${total_cost:.2f}**"
                                            
                                            st.success(recommendation)
                                            
                                        elif latest_prediction == 0 and latest_shares > 0:
                                            action = "SELL"
                                            
                                            if shares_per_trade == -1:
                                                shares_to_trade = latest_shares
                                                recommendation = f"Based on our Buy/Sell strategy prediction, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** (all shares) of {page} at approximately \${latest_price:.2f} per share."
                                            else:
                                                shares_to_trade = min(shares_per_trade, latest_shares)
                                                recommendation = f"Based on our Buy/Sell strategy prediction, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page} at approximately \${latest_price:.2f} per share."
                                            
                                            # Additional context
                                            if latest_shares > shares_to_trade and shares_per_trade != -1:
                                                recommendation += f" You currently have {latest_shares} shares in total."
                                            
                                            # Total return
                                            total_return = shares_to_trade * latest_price
                                            recommendation += f" Total return: **${total_return:.2f}**"
                                            
                                            st.warning(recommendation)
                                            
                                        else:
                                            action = "HOLD"
                                            if latest_prediction == 1:
                                                reason = f"insufficient cash (\${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                                            else:
                                                reason = f"you don't have any shares to sell (current shares: {latest_shares})"
                                            
                                            recommendation = f"Based on our Buy/Sell strategy analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                                            
                                            st.info(recommendation)
                                    
                                    elif strategy == "Buy the Dip":
                                        # Get relevant data
                                        latest_log_return = latest_data['Log_Return']
                                        
                                        if latest_log_return <= dip_threshold and latest_prediction == 1 and latest_cash >= latest_price:
                                            action = "BUY"
                                            
                                            if shares_per_trade == -1:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = max_shares
                                                recommendation = f"Based on our Buy the Dip strategy, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** (maximum affordable) of {page}. A dip of {latest_log_return:.2%} was detected (threshold: {dip_threshold:.2%}), and our model predicts an upcoming price increase."
                                            else:
                                                max_shares = int(latest_cash // latest_price)
                                                shares_to_trade = min(shares_per_trade, max_shares)
                                                recommendation = f"Based on our Buy the Dip strategy, we recommend: **{action} {shares_to_trade} share{'s' if shares_to_trade > 1 else ''}** of {page}. A dip of {latest_log_return:.2%} was detected (threshold: {dip_threshold:.2%}), and our model predicts an upcoming price increase."
                                            
                                            # Additional context
                                            if max_shares > shares_to_trade and shares_per_trade != -1:
                                                recommendation += f" You have enough cash (${latest_cash:.2f}) to purchase up to {max_shares} shares."
                                            
                                            # Total cost
                                            total_cost = shares_to_trade * latest_price
                                            recommendation += f" Total cost: **${total_cost:.2f}**"
                                            
                                            st.success(recommendation)
                                            
                                        else:
                                            action = "HOLD"
                                            if latest_log_return > dip_threshold:
                                                reason = f"no significant dip has been detected (current log return: {latest_log_return:.2%}, threshold: {dip_threshold:.2%})"
                                            elif latest_prediction != 1:
                                                reason = "our model doesn't predict a price increase yet"
                                            elif latest_cash < latest_price:
                                                reason = f"insufficient cash (\${latest_cash:.2f}) to purchase shares at current price (${latest_price:.2f})"
                                            else:
                                                reason = "current market conditions don't meet our trading criteria"
                                            
                                            recommendation = f"Based on our Buy the Dip strategy analysis, we recommend: **{action}** your current position of {page}. This is because {reason}."
                                            
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
                                    buy_points = portfolio_df[portfolio_df['Trade'].str.startswith('Buy')]
                                    if not buy_points.empty:
                                        fig1.add_trace(go.Scatter(
                                            x=buy_points['Date'],
                                            y=buy_points['Close'],
                                            mode='markers',
                                            name='Buy',
                                            marker=dict(color='green', size=10, symbol='triangle-up')
                                        ))

                                    # Add markers for sell points
                                    sell_points = portfolio_df[portfolio_df['Trade'].str.startswith('Sell')]
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

                                    # Log Returns Chart
                                    fig_log = go.Figure()

                                    # Add log returns trace
                                    fig_log.add_trace(go.Scatter(
                                        x=ml_df['Date'], y=ml_df['Log_Return'], mode='lines', name='Log Returns'
                                    ))

                                    # Add horizontal threshold line if using Buy the Dip strategy
                                    if strategy == "Buy the Dip":
                                        fig_log.add_trace(go.Scatter(
                                            x=[ml_df['Date'].min(), ml_df['Date'].max()],
                                            y=[dip_threshold, dip_threshold],
                                            mode='lines',
                                            name='Buy Next Day Threshold',
                                            line=dict(color='green', width=2, dash='dash')
                                        ))
                                        
                                        # Add annotation to the threshold line
                                        fig_log.add_annotation(
                                            x=ml_df['Date'].max(),
                                            y=dip_threshold,
                                            text="Buy Next Day Threshold",
                                            showarrow=False,
                                            yshift=10,
                                            font=dict(color="green"),
                                            bgcolor="rgba(255,255,255,0.8)"
                                        )

                                    fig_log.update_layout(
                                        title=f"{page} Log Returns with Buy Threshold", 
                                        xaxis_title='Date', 
                                        yaxis_title='Log Return'
                                    )
                                    st.plotly_chart(fig_log, use_container_width=True)

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
                                        
                                        - Buy when the model predicts the stock price will increase (Predicted_Target = 1)
                                        - Hold your current position when the model predicts a price decrease (Predicted_Target = 0)
                                        - Never sell shares once purchased
                                        
                                        This approach is similar to a traditional buy and hold strategy but uses ML signals to time purchases.
                                        """)
                                    elif strategy == "Buy & Sell":
                                        st.markdown("""
                                        **Buy & Sell Strategy**
                                        
                                        This strategy actively trades based on machine learning predictions:
                                        
                                        - Buy shares when the model predicts the stock price will increase (Predicted_Target = 1)
                                        - Sell shares when the model predicts the stock price will decrease (Predicted_Target = 0)
                                        - Each buy/sell decision is made independently each day based on the latest prediction
                                        
                                        This approach attempts to capitalize on both upward and downward price movements.
                                        """)
                                    elif strategy == "Buy the Dip":
                                        st.markdown(f"""
                                        **Buy the Dip Strategy**
                                        
                                        This strategy looks for significant dips in price before buying:
                                        
                                        - Identify a dip using log returns (threshold: {dip_threshold:.3f})
                                        - A negative value of {dip_threshold:.3f} means we look for price drops of approximately {-dip_threshold*100:.1f}%
                                        - If the price drops below this threshold, mark it as a dip
                                        - If the model predicts a price increase (Predicted_Target = 1) within the next **3 days**, buy shares
                                        - Hold if no buy signal appears after a dip
                                        - This strategy never sells shares once purchased
                                        
                                        This strategy aims to take advantage of short-term market pullbacks while using ML predictions for confirmation.
                                        """)
                                
                                # Display ML-ready data in expander
                                with st.expander("View ML-Ready Data"):
                                    st.dataframe(ml_df)
                                
                                # Display raw data in expander
                                with st.expander("View Raw Data"):
                                    st.dataframe(df)
                        except Exception as e:
                            st.error(f"❌ Error making predictions: {str(e)}")
                            st.info("There may be an issue with the data format or the machine learning model.")
                else:
                    st.error("❌ Failed to fetch any data. Please try again later.")
                    st.info("If the problem persists, try a different date range or check if the SimFin API is available.")
    else:
        st.info("👈 Select a stock ticker, date range, and trading strategy, then click 'Fetch Stock Data' to view the results")