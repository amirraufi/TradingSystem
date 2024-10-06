import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.Getting_data import get_data_dict_with_carry
from Calculators.std_Ret_calculator import calculate_variable_standard_deviation_for_risk_targeting_from_dict, calculate_position_series_given_variable_risk_for_dict, standardDeviation, aggregate_returns
from Strategies.Buffering import apply_buffering_to_position_dict
from Calculators.std_Ret_calculator import calculate_perc_returns_for_dict_with_costs
from Calculators.FX import create_fx_series_given_adjusted_prices_dict
from Calculators.Stats_Calculators import calculate_stats

def calculate_position_dict_with_multiple_carry_forecast_applied(
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    average_position_contracts_dict: dict,
    carry_prices_dict: dict,
    carry_spans: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_carry_forecast_applied(
                    average_position=average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                    carry_spans=carry_spans,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry

def calculate_position_with_multiple_carry_forecast_applied(
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    carry_spans: list,
) -> pd.Series:

    forecast = calculate_combined_carry_forecast(
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        carry_spans=carry_spans,
    )

    return forecast * average_position / 10


# Function to calculate the combined carry forecast
def calculate_combined_carry_forecast(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    carry_spans: list,
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast_for_carry(
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            span=span,
        )
        for span in carry_spans
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(carry_spans)
    FDM_DICT = {1: 1.0, 2: 1.02, 3: 1.03, 4: 1.04}
    fdm = FDM_DICT[rule_count]

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast

# Function to calculate forecast for a specific carry span
def calculate_forecast_for_carry(
    stdev_ann_perc: standardDeviation, carry_price: pd.DataFrame, span: int
):

    smooth_carry = calculate_smoothed_carry(
        stdev_ann_perc=stdev_ann_perc, carry_price=carry_price, span=span
    )
    scaled_carry = smooth_carry * 30
    capped_carry = scaled_carry.clip(-20, 20)
    return capped_carry

# Function to smooth carry data
def calculate_smoothed_carry(
    stdev_ann_perc: standardDeviation, carry_price: pd.DataFrame, span: int
):

    risk_adj_carry = calculate_vol_adjusted_carry(
        stdev_ann_perc=stdev_ann_perc, carry_price=carry_price
    )

    smooth_carry = risk_adj_carry.ewm(span).mean()

    return smooth_carry

# Function to adjust carry data by volatility
def calculate_vol_adjusted_carry(
    stdev_ann_perc: standardDeviation, carry_price: pd.DataFrame
) -> pd.Series:

    ann_carry = calculate_annualised_carry(carry_price)
    ann_price_vol = stdev_ann_perc.annual_risk_price_terms()

    risk_adj_carry = ann_carry.ffill() / ann_price_vol.ffill()

    return risk_adj_carry

# Function to calculate annualised carry
def calculate_annualised_carry(
    carry_price: pd.DataFrame,
):

    ## will be reversed if price_contract > carry_contract
    raw_carry = carry_price["PRICE"] - carry_price["CARRY"]
    contract_diff = _total_year_frac_from_contract_series(carry_price['CARRY_CONTRACT']) - _total_year_frac_from_contract_series(carry_price['PRICE_CONTRACT'])

    ann_carry = raw_carry / contract_diff

    return ann_carry

# Helper functions to calculate the year fraction from contract series
def _total_year_frac_from_contract_series(x):
    return _year_from_contract_series(x) + _month_as_year_frac_from_contract_series(x)

def _year_from_contract_series(x):
    return x // 10000

def _month_as_year_frac_from_contract_series(x):
    return (x % 10000) / 100 / 12
FDM_LIST = {
    1: 1.0, 2: 1.02, 3: 1.03, 4: 1.23, 5: 1.25,
    6: 1.27, 7: 1.29, 8: 1.32, 9: 1.34, 10: 1.35,
    11: 1.36, 12: 1.38, 13: 1.39, 14: 1.41, 15: 1.42,
    16: 1.44, 17: 1.46, 18: 1.48, 19: 1.50, 20: 1.53,
    21: 1.54, 22: 1.55, 25: 1.69, 30: 1.81, 35: 1.93, 40: 2.00
}
fdm_x = list(FDM_LIST.keys())
fdm_y = list(FDM_LIST.values())
f_interp = interp1d(fdm_x, fdm_y, bounds_error=False, fill_value=2)

def get_fdm(rule_count):
    return float(f_interp(rule_count))



# if __name__ == "__main__":
#     # Fetch data for all instruments, but filter for SP500 and Gold
#     adjusted_prices_dict, current_prices_dict, carry_prices_dict = get_data_dict_with_carry()

#     # Filtered dictionaries for SP500 and Gold only
#     filtered_instruments = ['SP500', 'Gold']
#     adjusted_prices_dict = {k: v for k, v in adjusted_prices_dict.items() if k in filtered_instruments}
#     current_prices_dict = {k: v for k, v in current_prices_dict.items() if k in filtered_instruments}
#     carry_prices_dict = {k: v for k, v in carry_prices_dict.items() if k in filtered_instruments}

#     # Parameters specific to SP500 and Gold
#     multipliers = dict(SP500=50, Gold=1000)
#     risk_target_tau = 0.50
#     fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

#     capital = 20000000

#     idm = 1.5
#     instrument_weights = dict(SP500=0.5, Gold=0.5)
#     cost_per_contract_dict = dict(SP500=0.875, Gold=1.3)

#     # Calculate standard deviations
#     std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
#         adjusted_prices=adjusted_prices_dict, current_prices=current_prices_dict
#     )

#     # Calculate the average position contracts for the filtered instruments
#     average_position_contracts_dict = calculate_position_series_given_variable_risk_for_dict(
#         capital=capital,
#         risk_target_tau=risk_target_tau,
#         idm=idm,
#         weights=instrument_weights,
#         std_dev_dict=std_dev_dict,
#         fx_series_dict=fx_series_dict,
#         multipliers=multipliers,
#     )

#     # Carry forecast spans
#     carry_spans = [5, 20, 60, 120]

#     # Calculate positions with multiple carry forecasts applied
#     position_contracts_dict = calculate_position_dict_with_multiple_carry_forecast_applied(
#         adjusted_prices_dict=adjusted_prices_dict,
#         carry_prices_dict=carry_prices_dict,
#         std_dev_dict=std_dev_dict,
#         average_position_contracts_dict=average_position_contracts_dict,
#         carry_spans=carry_spans,
#     )

#     # Apply buffering to position contracts
#     buffered_position_dict = apply_buffering_to_position_dict(
#         position_contracts_dict=position_contracts_dict,
#         average_position_contracts_dict=average_position_contracts_dict,
#     )

#     # Calculate percentage returns with costs for SP500 and Gold
#     perc_return_dict = calculate_perc_returns_for_dict_with_costs(
#         position_contracts_dict=buffered_position_dict,
#         fx_series=fx_series_dict,
#         multipliers=multipliers,
#         capital=capital,
#         adjusted_prices=adjusted_prices_dict,
#         cost_per_contract_dict=cost_per_contract_dict,
#         std_dev_dict=std_dev_dict,
#     )

#     # Example output of percentage returns
#     print("Percentage returns for SP500:", perc_return_dict['SP500'].head())
#     print("Percentage returns for Gold:", perc_return_dict['Gold'].head())

# def plot_returns(returns_dict: dict):
#     """
#     Function to plot percentage returns for each instrument in the returns_dict.

#     Args:
#     - returns_dict (dict): A dictionary where keys are instrument names (e.g., 'SP500', 'Gold')
#                            and values are pandas Series containing percentage returns.
#     """
#     plt.figure(figsize=(10, 6))

#     for instrument, returns in returns_dict.items():
#         # Check if the returns are valid (not a float)
#         if isinstance(returns, pd.Series):
#             # Plot returns for each instrument
#             plt.plot(returns.index, returns, linestyle='-', label=instrument)
    
#     plt.title('Percentage Returns')
#     plt.xlabel('Index')
#     plt.ylabel('Returns')
#     plt.legend()
#     plt.grid(True)
#     plt.show()




# # Then plot the cleaned returns
# stats_sp500 = calculate_stats(perc_return_dict['SP500'])
# print("SP500 Stats:", stats_sp500)

# stats_gold = calculate_stats(perc_return_dict['Gold'])
# print("Gold Stats:", stats_gold)


# perc_return_agg = aggregate_returns(perc_return_dict)
# print(calculate_stats(perc_return_agg))