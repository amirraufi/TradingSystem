from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import time
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from scipy.stats import norm
from enum import Enum
import matplotlib
matplotlib.use("TkAgg")


instrument_mapping = {
    "SP500": {
        "yf": "ES=F",  # S&P 500 Futures
        "ib": {"symbol": "ES", "exchange": "CME"}
    },
    "gas": {
        "yf": "NG=F",  # Natural Gas Futures
        "ib": {"symbol": "NG", "exchange": "NYMEX"}
    },
    "oil": {
        "yf": "CL=F",  # Crude Oil Futures
        "ib": {"symbol": "CL", "exchange": "NYMEX"}
    },
    "gold": {
        "yf": "GC=F",  # Gold Futures
        "ib": {"symbol": "GC", "exchange": "COMEX"}
    },
    "US5": {
        "yf": "FV=F",  # 5-Year Treasury Futures
        "ib": {"symbol": "FV", "exchange": "CBOT"}
    },
    "US10": {
        "yf": "ZN=F",  # 10-Year Treasury Futures
        "ib": {"symbol": "ZN", "exchange": "CBOT"}
    },
    "US2": {
        "yf": "ZT=F",  # 2-Year Treasury Futures
        "ib": {"symbol": "ZT", "exchange": "CBOT"}
    },
    "vix": {
        "yf": "^VIX",  # VIX Index
        "ib": {"symbol": "VIX", "exchange": "CBOE"}
    },
    "usdgbp": {
        "yf": "GBPUSD=X",  # USD/GBP currency pair
        "ib": {"symbol": "GBP", "exchange": "IDEALPRO"}  # Currency, no expiry
    }
}
SP_500_RC_data = pd.read_csv("Data/SP500_daily.csv")
VIX_RC_data = pd.read_csv("Data/VIX_daily.csv")
US10_RC_data = pd.read_csv("Data/US10_daily.csv")
Gold_RC_data = pd.read_csv("Data/GOLD_daily.csv")
GBPEUR_data = pd.read_csv("Data/GBPEUR_daily.csv")


# Create a class to handle the API connection and responses
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize list to store data

    def historicalData(self, reqId, bar):
        print(f"Date: {bar.date}, Close: {bar.close}, Volume:{bar.volume}")
        self.data.append([bar.date, bar.close, bar.volume])  # Store data

    def historicalDataEnd(self, reqId, start, end):
        print("Historical data download complete")
        self.disconnect()  # Disconnect when data download is complete

def request_contract_details(symbol, exchange, expiration_date):
    app = IBapi()
    app.connect("127.0.0.1", 7496, 123)  # Paper trading port (7497) or live trading port (7496)
    time.sleep(1)  # Sleep to allow connection to establish

    contract = create_contract(symbol, exchange, expiration_date)
    app.reqContractDetails(1, contract)

    # Keep the connection open until data is retrieved
    app.run()
# Function to create contract for futures
def create_contract(symbol, exchange, expiration_date):
    contract = Contract()
    contract.symbol = symbol  # E.g., "GC" for gold futures
    contract.secType = "FUT"  # Futures contract
    contract.exchange = exchange  # E.g., "COMEX" for gold
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = expiration_date  # Expiration date
    return contract

# Function to connect and retrieve historical data
def getting_futures_data(symbol, exchange, expiration_date):
    app = IBapi()
    app.connect("127.0.0.1", 7496, 123)  # Connect to TWS (or 7496 for live trading)
    
    time.sleep(1)  # Sleep to allow connection to establish

    # Create contract and request historical data
    contract = create_contract(symbol, exchange, expiration_date)
    app.reqHistoricalData(1, contract, "", "1 Y", "1 day", "TRADES", 1, 1, False, [],)
    
    # Keep the connection open until data is retrieved
    app.run()

    # Convert the data into a pandas DataFrame once disconnected
    df = pd.DataFrame(app.data, columns=["Date", "Close", "Volume",])
    
    # Save the data to CSV or return the DataFrame
    # df.to_csv("/Users/amirhossein/Desktop/gold_futures.csv", index=False)
    return df

# Main block to call the function
# if __name__ == "__main__":
#     symbol = "GC"
#     exchange = "COMEX"
#     expiration_date = "202410"  # E.g., December 2025
#     data_future = getting_futures_data(symbol, exchange, expiration_date)
#     print(data_future.head())
# Now we will go for the yfinance datas.

tickers = {
    "Gold" : "GC=F",
    "Treasury" : "ZN=F",
    "GBPUSD" : "GBPUSD=X",
    "S&P500" :"ES=F",
    "VIX" : "^VIX"
}
today_date = date.today()
def download_data_yf(ticker, start_date = "2004-01-01", end_date=today_date):
    data = yf.download(ticker, start= start_date, end= end_date)
    data.dropna(inplace=True)
    return data
def returns(x):
    x["returns"] = np.log(x["Close"] / x["Close"].shift(1))

def get_hourly_futures_data(symbol,exchange, expiration_date):
    app = IBapi
    app.connect("127.0.0.1", 7496, 123)
    time.sleep(1)
    Contract = create_contract(symbol=symbol, exchange=exchange, expiration_date=expiration_date)
    app.reqHistoricalData(1, Contract,"","20 Y","1 hour","Trades",1,1, False, [])
    app.run
    df = pd.DataFrame(app.data, columns=["Date", "Close", "Volume"])
    return df

# Load data from Yahoo Finance and IB (this assumes you've written the download and getting functions)


# Enum for Frequency
Frequency = Enum("Frequency", "Natural Year Month Week BDay")
NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week

# Function to load data
def load_data(instrument, expiration_date):
    yf_ticker = instrument_mapping[instrument]["yf"]
    ib_ticker = instrument_mapping[instrument]["ib"]["symbol"]
    ib_exchange = instrument_mapping[instrument]["ib"]["exchange"]
    
    # Load data using Yahoo Finance and IBAPI
    yf_data = download_data_yf(yf_ticker)  # Data from Yahoo Finance
    ib_data = getting_futures_data(symbol=ib_ticker, exchange=ib_exchange, expiration_date=expiration_date)
    
    return yf_data, ib_data


# Function to adjust prices between Yahoo Finance and IB data
def adj_prices(instrument, expiration_date):
    yf_data, ib_data = load_data(instrument=instrument, expiration_date=expiration_date)
    difference = yf_data["Close"].iloc[-1] - ib_data["Close"].iloc[-1]
    if difference > ib_data["Close"].iloc[-1] * 0.005:
        yf_data["Close"] += difference  # Adjust YF data based on IB data
    return yf_data

# Function to fetch data from Yahoo Finance for the instruments
def get_data_dict(instrument_list, expiration_dates):
    adjusted_prices = {}
    
    # Load data from Yahoo Finance
    all_data = {instrument: download_data_yf(instrument_mapping[instrument]["yf"]) for instrument in instrument_list}
    
    # Fetch adjusted prices using the adj_prices function for each instrument and its corresponding expiration date
    for instrument, expiration_date in zip(instrument_list, expiration_dates):
        # Adjust prices based on IB data (if applicable)
        adj_data = adj_prices(instrument=instrument, expiration_date=expiration_date)
        
        # Store adjusted prices and current prices
        adjusted_prices[instrument] = adj_data["Close"]
    
    # Extract current prices from downloaded data
    current_prices = {instrument: data["Close"] for instrument, data in all_data.items()}
    
    return adjusted_prices, current_prices

        

# Function to fetch both carry and adjusted prices from YF and IB


# Function to get hourly and daily adjusted prices directly from IB
def get_data_dict_with_hourly_adjusted(instrument_list, expiration_dates):
    # Fetch hourly data using IBAPI
    adjusted_prices_hourly_dict = {
        instrument_code: get_hourly_futures_data(
            symbol=instrument_mapping[instrument_code]["ib"], 
            exchange=instrument_mapping[instrument_code]["exchange"], 
            expiration_date=expiration_dates[instrument_code]
        )
        for instrument_code in instrument_list
    }

    # Fetch daily data using IBAPI
    adjusted_prices_daily_dict = {
        instrument_code: getting_futures_data(
            symbol=instrument_mapping[instrument_code]["ib"], 
            exchange=instrument_mapping[instrument_code]["exchange"], 
            expiration_date=expiration_dates[instrument_code]
        )
        for instrument_code in instrument_list
    }

    # Extract the daily closing prices as current prices
    current_prices_daily_dict = {
        instrument_code: adjusted_prices_daily_dict[instrument_code]["Close"]
        for instrument_code in instrument_list
    }

    return adjusted_prices_hourly_dict, adjusted_prices_daily_dict, current_prices_daily_dict

def last_friday(year, month):
    last_day = datetime(year, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(year, 12, 31)
    last_friday = last_day - timedelta(days=(last_day.weekday() - 4 + 7) % 7)
    return last_friday



# Function to get the data dictionary with carry
def get_data_dict_with_carry(instrument_list: list = None):
    # Set default instrument list if not provided
    if instrument_list is None:
        instrument_list = ['SP500', 'VIX', 'US10', 'Gold', 'GBPEUR']

    # Load the main data (adjusted and current prices)
    all_data = dict(
        [
            (instrument_code, pd.read_csv(f"Data/{instrument_code}_daily.csv"))  # Adjust the path as needed
            for instrument_code in instrument_list
        ]
    )

    # Extract the adjusted prices from the loaded data
    adjusted_prices = dict(
        [
            (instrument_code, data_for_instrument['FORWARD'])  # Assuming 'FORWARD' is the column name
            for instrument_code, data_for_instrument in all_data.items()
        ]
    )

    # Extract the current prices (e.g., underlying prices) from the loaded data
    current_prices = dict(
        [
            (instrument_code, data_for_instrument['PRICE'])  # Assuming 'PRICE' is the column name
            for instrument_code, data_for_instrument in all_data.items()
        ]
    )

    # Load the carry data from separate CSV files
    carry_data = dict(
        [
            (instrument_code, pd.read_csv(f"Data/{instrument_code}_daily.csv"))  # Assuming carry data has separate files
            for instrument_code in instrument_list
        ]
    )

    return adjusted_prices, current_prices, carry_data
#  an example
adjusted_prices, current_prices, carry_data = get_data_dict_with_carry()

# Example access:
print(adjusted_prices['SP500'].head())
print(carry_data['Gold'].head())




# Dummy function to simulate IB API fetching - Replace with actual IB fetching logic
# def get_ib_data(symbol, exchange, expiration_date):
#     app = IBapi()
#     app.connect("127.0.0.1", 7496, 123)  # Paper trading port (7497) or live trading port (7496)
    
#     time.sleep(1)  # Sleep to allow connection to establish

#     # Create the futures contract and request historical data
#     contract = create_contract(symbol, exchange, expiration_date)
#     app.reqHistoricalData(1, contract, "", "1 M", "1 day", "TRADES", 1, 1, False, [])

#     # Keep the connection open until data is retrieved
#     app.run()

#     # Convert the data into a pandas DataFrame once disconnected
#     df = pd.DataFrame(app.data, columns=["Date", "Close"])
#     return df

# # Main function to fetch raw price data from IB with roll logic
# def fetch_daily_price_data_with_roll(symbol, exchange, expiration_dates):
#     data = pd.DataFrame()
#     current_date = pd.to_datetime("2023-09-01")
#     today = datetime.now()

#     while current_date <= today and len(expiration_dates) > 1:
#         # Pop the first (nearest) expiry for each iteration
#         near_expiry = expiration_dates.pop(0)
#         far_expiry = expiration_dates[0]  # After popping, the new first element becomes the far expiry

#         near_year, near_month = int(near_expiry[:4]), int(near_expiry[4:])
#         far_year, far_month = int(far_expiry[:4]), int(far_expiry[4:])

#         # Get last Friday for near contract
#         last_friday_near = last_friday(near_year, near_month)

#         # Fetch near and far contract data from IB
#         near_data = get_ib_data(symbol, exchange, near_expiry)
#         far_data = get_ib_data(symbol, exchange, far_expiry)

#         # Merge the two datasets on the Date column
#         merged_data = pd.merge(near_data, far_data, on='Date', suffixes=('_near', '_far'))

#         # Filter data until the last Friday of the near contract's expiration month
#         filtered_data = merged_data[pd.to_datetime(merged_data['Date']) <= last_friday_near]

#         # Append to the overall dataset
#         data = pd.concat([data, filtered_data], ignore_index=True)

#         # Move current_date to the last Friday of the near contract's expiration month
#         current_date = last_friday_near + timedelta(days=1)
#         if current_date > today:
#             break

#     return data
# expiration_dates = ["202309", "202312", "202403", "202406", "202409", "202412"]
# symbol = "GC"  # Example for Treasury bond
# exchange = "COMEX"  # Example exchange

# price_data = fetch_daily_price_data_with_roll(symbol, exchange, expiration_dates)

# # Rename columns to match your CSV structure
# price_data.rename(columns={
#     'Date': 'date',
#     'Close_near': 'near_price',
#     'Close_far': 'far_price'
# }, inplace=True)

# print(price_data.head())
# def getting_ex_futures_data(symbol, exchange, expiration_date):
#     app = IBapi()
#     app.connect("127.0.0.1", 7496, 123)  # Connect to TWS (or 7496 for live trading)
    
#     time.sleep(1)  # Sleep to allow connection to establish

#     # Create contract and request historical data
#     contract = create_contract(symbol, exchange, expiration_date)
#     app.reqHistoricalData(1, contract, "20240920-23:59:59", "1 M", "1 day", "TRADES", 1, 1, False, [],)
    
#     # Keep the connection open until data is retrieved
#     app.run()

#     # Convert the data into a pandas DataFrame once disconnected
#     df = pd.DataFrame(app.data, columns=["Date", "Close", "Volume",])
    
#     # Save the data to CSV or return the DataFrame
#     # df.to_csv("/Users/amirhossein/Desktop/gold_futures.csv", index=False)
#     return df
# getting_ex_futures_data("GC", "CBOE", "202409")