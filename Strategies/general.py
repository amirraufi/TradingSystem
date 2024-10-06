import numpy as np
import pandas as pd 
from sklearn.utils import resample
from scipy.optimize import minimize
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Calculators.std_Ret_calculator import standardDeviation

def bootstrap_correlation(data, n_iterations=1000, subset_size=0.1):
    correlation_matrices = []

    # Perform bootstrapping
    for i in range(n_iterations):
        # Randomly sample a subset of the data (with replacement)
        bootstrap_sample = resample(data, n_samples=int(len(data) * subset_size))
        
        # Calculate correlation matrix for the subset
        corr_matrix = bootstrap_sample.corr()
        correlation_matrices.append(corr_matrix)

    # Calculate the mean correlation matrix across all bootstrap samples
    avg_correlation = np.mean(correlation_matrices, axis=0)
    
    return avg_correlation

def objective_function(weight, excess_ret, p, alpha):
    diversification = np.dot(np.dot(weight.T,p),weight)
    portfolio_return = np.dot(excess_ret, weight)
    portfolio_std = np.std(portfolio_return)
    SR = portfolio_return/portfolio_std if portfolio_std !=0 else 0
    return alpha * diversification +(1-alpha)* SR

def optimise_weights(excess_ret, p, alpha=0.5):
    num_assets = excess_ret.shape[1]
    # number of the assets or strategies
    initial_weights = np.ones(num_assets)/num_assets
    constraints = ({"Type": "eq", "fun": lambda weights: np.sum(weights)-1})
    bounds = [(0,1) for _ in range(num_assets)]
    result = minimize(objective_function, initial_weights, args=(excess_ret,p, alpha), method="SLSQP", bounds= bounds, constraints=constraints)
    return result.x
def bootstrap_optimise_weights(excess_ret, num_samples= 100, num_runs=1000, alpha =0.5):
    all_weights = []
    for i in range(num_runs):
        sample_data = excess_ret.sample(n=num_samples, replace= True)
        p = bootstrap_correlation(excess_ret)
        optimised_weights = optimise_weights(sample_data, p, alpha)
        all_weights.append(optimise_weights)
    all_weights= np.array(all_weights)
    avg_weight = np.mean(all_weights, axis=0)
    return avg_weight
# using the panama method for a different expiration than our front month 

def Back_adj_prices(yf_prices, Future_price):
    """
    We want to adjust the yahoo finance prices using the future prices that we are targetting using the panama methode
    """
    yf_closing_prices = yf_prices["Close"]
    Future_closing_price= Future_price["Close"]
    latest_yf_close = yf_closing_prices.iloc[-1]
    latest_future_close = Future_closing_price.iloc[-1]
    if abs(latest_future_close-latest_yf_close)<0.005*latest_future_close:
        adj_prices = yf_closing_prices
    else:
        adj_factor = latest_future_close - latest_yf_close
        adj_prices = yf_closing_prices.copy()
        adj_prices+= adj_factor
    return adj_prices

def calculate_normalised_price_dict(adjusted_prices_dict: dict, std_dev_dict: dict) -> dict:
    normalised_price_dict = {
        instrument: calculate_normalised_price(
            adjusted_price=adjusted_prices_dict[instrument],
            instrument_risk=std_dev_dict[instrument]
        )
        for instrument in adjusted_prices_dict.keys()
    }
    return normalised_price_dict

def calculate_normalised_price(adjusted_price: pd.Series, instrument_risk: standardDeviation) -> pd.Series:
    # Risk-adjusted daily price terms
    daily_price_instrument_risk = instrument_risk.daily_risk_price_terms()
    
    # Calculating the normalised returns (percentage change relative to risk)
    normalised_returns = 100 * (adjusted_price.diff() / daily_price_instrument_risk)
    
    # Handle NaN values (set to 0)
    normalised_returns[normalised_returns.isna()] = 0
    
    # Cumulative sum of the normalised returns
    normalised_price = normalised_returns.cumsum()
    
    return normalised_price

def _total_year_frac_from_contract_series(x):
    return _year_from_contract_series(x) + _month_as_year_frac_from_contract_series(x)

def _year_from_contract_series(x):
    return x // 10000

def _month_as_year_frac_from_contract_series(x):
    return (x % 10000) / 100 / 12

    
    
    


    
    
    
        
    
