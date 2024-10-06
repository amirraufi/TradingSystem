import pandas as pd 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Strategies.Trend_following_n import calculate_forecast_for_ewmac
from Calculators.FX import create_fx_series_given_adjusted_prices_dict
from Calculators.std_Ret_calculator import calculate_variable_standard_deviation_for_risk_targeting_from_dict, calculate_position_series_given_variable_risk_for_dict, calculate_perc_returns_for_dict_with_costs, standardDeviation
from Strategies.Buffering import apply_buffering_to_position_dict
from Data.Getting_data import get_data_dict
from Calculators.Stats_Calculators import calculate_stats
def calculate_position_dict_with_multiple_trend_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    fast_spans: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_trend_filter = dict(
        [
            (
                instrument_code,
                calculate_position_with_multiple_trend_forecast_applied(
                    adjusted_prices_dict[instrument_code],
                    average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    fast_spans=fast_spans,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_trend_filter


def calculate_position_with_multiple_trend_forecast_applied(
    adjusted_price: pd.Series,
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_spans: list,
) -> pd.Series:

    forecast = calculate_combined_ewmac_forecast(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        fast_spans=fast_spans,
    )

    return forecast * average_position / 10


def calculate_combined_ewmac_forecast(
    adjusted_price: pd.Series,
    stdev_ann_perc: standardDeviation,
    fast_spans: list,
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast_for_ewmac(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )
        for fast_span in fast_spans
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(fast_spans)
    FDM_DICT = {1: 1.0, 2: 1.03, 3: 1.08, 4: 1.13, 5: 1.19, 6: 1.26}
    fdm = FDM_DICT[rule_count]

    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast
if __name__ == "__main__":
    instrument_list = ["SP500","US10"]
    expiration_date = ["202412","202412"]
    adjusted_prices_dict, current_prices_dict = get_data_dict(instrument_list=instrument_list,expiration_dates=expiration_date)
    multipliers = dict(SP500=5, US10=1000)
    risk_target_tau = 2
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 100000000

    idm = 1.5
    instrument_weights = dict(SP500=0.5, US10=0.5)
    cost_per_contract_dict = dict(SP500=0.875, US10=5)

    std_dev_dict = calculate_variable_standard_deviation_for_risk_targeting_from_dict(
        adjusted_prices=adjusted_prices_dict, current_prices=current_prices_dict
    )

    average_position_contracts_dict = (
        calculate_position_series_given_variable_risk_for_dict(
            capital=capital,
            risk_target_tau=risk_target_tau,
            idm=idm,
            weights=instrument_weights,
            std_dev_dict=std_dev_dict,
            fx_series_dict=fx_series_dict,
            multipliers=multipliers,
        )
    )

    ## We use three arbitrary slow spans here for both instruments
    ## In reality we would need to check costs and turnover
    fast_spans = [16, 32, 64]
    position_contracts_dict = (
        calculate_position_dict_with_multiple_trend_forecast_applied(
            adjusted_prices_dict=adjusted_prices_dict,
            average_position_contracts_dict=average_position_contracts_dict,
            std_dev_dict=std_dev_dict,
            fast_spans=fast_spans,
        )
    )

    buffered_position_dict = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    print(calculate_stats(perc_return_dict["SP500"]))