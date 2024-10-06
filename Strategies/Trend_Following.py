import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Strategies.general import Std_in_currency
import numpy as np
from Strategies.general import Normalised_prices
from Strategies.general import Back_adj_prices
from Data.Getting_data import download_data_yf
import pandas as pd
# This is a divergent Strategy
def ewma(data, window):
    # Ensure the data is numeric; coerce non-numeric to NaN
    data_numeric = pd.to_numeric(data, errors='coerce')
    return data_numeric.ewm(span=window, adjust=False).mean()
def exponential_momentum_generator(x, span):
    # Apply EWMA only on numeric data
    x[f'EWMA_{span}'] = ewma(x, span)
def EWMA_Signals(x, span1, span2):
    # Ensure that the indices of the two EWMA Series are aligned before comparison
    ewma_span1 = x[f'EWMA_{span1}'].reindex_like(x[f'EWMA_{span2}'])
    ewma_span2 = x[f'EWMA_{span2}']
    
    # Compare the aligned Series
    x[f'Signal_{span1}_{span2}'] = np.where(ewma_span1 > ewma_span2, 1, -1)


def EWMA_Signal_strength_Scaled_forecast(x, std, price, span1=16, span2=64):
    std_price = Std_in_currency(std, price)
    x[f"row_Signal_{span1}_{span2}_S"] = (ewma(x, span1) - ewma(x, span2)) / std_price
    avg_row_forecast = np.average(x[f"row_Signal_{span1}_{span2}_S"])
    if avg_row_forecast != 0:
        scalar = 10 / avg_row_forecast
    else:
        scalar = 1  
    x[f"Scaled_Signal_{span1}_{span2}_S"]= (x[f"row_Signal_{span1}_{span2}_S"]* scalar).clip(-20,20)

    
def EWMA_Normalised_Trend(x, span1=16, span2=64):
    # Calculate the normalised prices
    normalised_prices = Normalised_prices(x)
    
    # Check if the data contains valid numeric types
    normalised_prices = pd.to_numeric(normalised_prices, errors='coerce')
    
    # Drop NaN values if they exist
    normalised_prices = normalised_prices.dropna()

    # Calculate the standard deviation
    norm_prices_std = normalised_prices.std()
    
    # Make sure we have valid standard deviation values
    if pd.isna(norm_prices_std):
        norm_prices_std = 1  # Default to 1 if the std is invalid
    
    # Perform EWMA and signals calculations
    exponential_momentum_generator(normalised_prices, span1)
    exponential_momentum_generator(normalised_prices, span2)
    EWMA_Signals(normalised_prices, span1, span2)

    # Return the final forecast
    return EWMA_Signal_strength_Scaled_forecast(normalised_prices, norm_prices_std, normalised_prices, span1, span2)

# Now call your function
gold_data = download_data_yf("GC=F")
gold_price = gold_data["Close"]
EWMA_Normalised_Trend(gold_price)


    
    
    