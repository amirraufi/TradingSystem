from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import time
from datetime import datetime

# Wrapper and client for IB API
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Store historical data

    def historicalData(self, reqId, bar):
        print(f"Date: {bar.date}, Close: {bar.close}")
        self.data.append([bar.date, bar.close])

    def historicalDataEnd(self, reqId, start, end):
        print("Historical data download complete")
        self.disconnect()

# Function to create an FX contract
def create_fx_contract(currency: str) -> Contract:
    contract = Contract()
    contract.symbol = currency
    contract.secType = "CASH"  # Spot FX
    contract.exchange = "IDEALPRO"  # Exchange for FX
    contract.currency = "USD"  # We want prices relative to USD
    return contract

# Function to fetch FX prices from IB
def get_fx_prices(currency: str) -> pd.Series:
    app = IBapi()
    app.connect("127.0.0.1", 7496, 123)  # Use 7497 for paper trading

    time.sleep(1)  # Ensure connection is established

    # Create FX contract (assuming USD base for simplicity)
    contract = create_fx_contract(currency)
    
    # Request historical data for 1 year, daily prices
    app.reqHistoricalData(1, contract, "", "1 Y", "1 day", "MIDPOINT", 1, 1, False, [])
    
    # Run the API until data is retrieved
    app.run()

    # Convert data into DataFrame
    df = pd.DataFrame(app.data, columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    
    # Ensure the format is a Pandas Series
    return df.squeeze()



def create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict: dict) -> dict:
    fx_series_dict = dict(
        [
            (
                instrument_code,
                create_fx_series_given_adjusted_prices(
                    instrument_code, adjusted_prices
                ),
            )
            for instrument_code, adjusted_prices in adjusted_prices_dict.items()
        ]
    )
    return fx_series_dict

def create_fx_series_given_adjusted_prices(
    instrument_code: str, adjusted_prices: pd.Series
) -> pd.Series:

    # Retrieve the currency based on the instrument code
    currency_for_instrument = fx_dict.get(instrument_code, "USD")
    if currency_for_instrument == "USD":
        return pd.Series(1, index=adjusted_prices.index)  # FX rate of 1 for USD

    # Fetch FX prices using IB
    fx_prices = get_fx_prices(currency_for_instrument)
    fx_prices_aligned = fx_prices.reindex(adjusted_prices.index).ffill()

    return fx_prices_aligned
# Expanding the dictionary for 100+ instruments across multiple asset classes
fx_dict = {
    # Stock indices
    "SP500": "USD",
    "NASDAQ": "USD",
    "nikkei": "JPY",
    "ftse100": "GBP",
    "DAX": "EUR",
    "cac40": "EUR",
    "hangseng": "HKD",
    "australia200": "AUD",
    "eurstx": "EUR",
    
    # Commodities
    "Gold": "USD",
    "Silver": "USD",
    "Crude_Oil": "USD",
    "natural_gas": "USD",
    "brent_oil": "USD",
    "platinum": "USD",
    "palladium": "USD",
    
    # Bonds and rates
    "US10": "USD",
    "US30": "USD",
    "bund": "EUR",
    "jgb": "JPY",
    
    # FX pairs
    "eurusd": "EUR",
    "gbpusd": "GBP",
    "usdjpy": "JPY",
    "usdcad": "CAD",
    "audusd": "AUD",
    "usdchf": "CHF",
    "eurjpy": "EUR",
    "gbpjpy": "GBP",
    "GBPEUR": "GBP",
    
    # Emerging markets
    "usdbrl": "BRL",
    "usdinr": "INR",
    "usdzar": "ZAR",
    "usdmxn": "MXN",
    
    # Cryptocurrencies 
    "bitcoin": "BTC",
    "ethereum": "ETH",
    
#    Volatility indexes
    "Vix": "USD",  # VIX Index
    "vstoxx": "EUR"
}

# You can now use this expanded `fx_dict` with the `create_fx_series_given_adjusted_prices` function.
