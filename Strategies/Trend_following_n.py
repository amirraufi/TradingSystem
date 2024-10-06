import pandas as pd 
import sys
import copy
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Calculators.std_Ret_calculator import standardDeviation

def calculate_position_dict_with_trend_filter_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_trend_filter_applied(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_trend_filter_applied(
    adjusted_price: pd.Series, average_position: pd.Series
) -> pd.Series:

    filtered_position = copy(average_position)
    ewmac_values = ewmac(adjusted_price)
    bearish = ewmac_values < 0
    filtered_position[bearish] = 0

    return filtered_position

def ewmac(adjusted_price: pd.Series, fast_span=16, slow_span=64) -> pd.Series:

    slow_ewma = adjusted_price.ewm(span=slow_span, min_periods=2).mean()
    fast_ewma = adjusted_price.ewm(span=fast_span, min_periods=2).mean()

    return fast_ewma - slow_ewma
def calculate_position_dict_with_symmetric_trend_filter_applied(adjusted_prices_dict, average_position_contracts_dict):
    position_dict_with_trend_filter = {
        instrument: calculate_position_with_symmetric_trend_filter_applied(
            adjusted_prices_dict[instrument], average_position_contracts_dict[instrument]
        )
        for instrument in adjusted_prices_dict.keys()
    }
    return position_dict_with_trend_filter

def calculate_position_with_symmetric_trend_filter_applied(adjusted_price, average_position):
    filtered_position = average_position.copy()
    ewmac_values = ewmac(adjusted_price)
    bearish = ewmac_values < 0
    filtered_position[bearish] = -filtered_position[bearish]
    return filtered_position

def calculate_forecast_for_ewmac(
    adjusted_price: pd.Series, stdev_ann_perc: standardDeviation, fast_span: int = 64
):

    scaled_ewmac = calculate_scaled_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    capped_ewmac = scaled_ewmac.clip(-20, 20)

    return capped_ewmac


def calculate_risk_adjusted_forecast_for_ewmac(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
):

    ewmac_values = ewmac(adjusted_price, fast_span=fast_span, slow_span=fast_span * 4)
    daily_price_vol = stdev_ann_perc.daily_risk_price_terms()

    risk_adjusted_ewmac = ewmac_values / daily_price_vol

    return risk_adjusted_ewmac

def calculate_scaled_forecast_for_ewmac(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
):

    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}
    risk_adjusted_ewmac = calculate_risk_adjusted_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )
    forecast_scalar = scalar_dict[fast_span]
    scaled_ewmac = risk_adjusted_ewmac * forecast_scalar

    return scaled_ewmac

def calculate_forecast_for_ewmac(adjusted_price, stdev_ann_perc, fast_span=64):
    scaled_ewmac = calculate_scaled_forecast_for_ewmac(adjusted_price=adjusted_price, stdev_ann_perc=stdev_ann_perc, fast_span=fast_span)
    capped_ewmac = scaled_ewmac.clip(-20, 20)  # Capping the forecast values
    return capped_ewmac

def calculate_position_dict_with_trend_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_span: int = 64,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_trend_forecast_applied(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    fast_span=fast_span,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_trend_forecast_applied(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_span: int = 64,
) -> pd.Series:

    forecast = calculate_forecast_for_ewmac(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_span=fast_span,
    )

    return forecast * average_position / 10

