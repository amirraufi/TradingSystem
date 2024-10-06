import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Strategies.Trend_following_n import calculate_forecast_for_ewmac
from Strategies.Carry import calculate_forecast_for_carry, get_data_dict_with_carry
from Calculators.FX import create_fx_series_given_adjusted_prices_dict
from Strategies.Buffering import apply_buffering_to_position_dict
from scipy.interpolate import interp1d
from Calculators.Stats_Calculators import calculate_stats
from Calculators.std_Ret_calculator import calculate_variable_standard_deviation_for_risk_targeting_from_dict, calculate_position_series_given_variable_risk_for_dict, aggregate_returns, calculate_perc_returns_for_dict_with_costs, standardDeviation
def calculate_position_dict_with_forecast_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule_spec: list,
) -> dict:

    list_of_instruments = list(adjusted_prices_dict.keys())
    position_dict_with_carry = dict(
        [
            (
                instrument_code,
                calculate_position_with_forecast_applied(
                    average_position=average_position_contracts_dict[instrument_code],
                    stdev_ann_perc=std_dev_dict[instrument_code],
                    carry_price=carry_prices_dict[instrument_code],
                    adjusted_price=adjusted_prices_dict[instrument_code],
                    rule_spec=rule_spec,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return position_dict_with_carry


def calculate_position_with_forecast_applied(
    average_position: pd.Series,
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:

    forecast = calculate_combined_forecast(
        adjusted_price=adjusted_price,
        stdev_ann_perc=stdev_ann_perc,
        carry_price=carry_price,
        rule_spec=rule_spec,
    )

    return forecast * average_position / 10


def calculate_combined_forecast(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule_spec: list,
) -> pd.Series:

    all_forecasts_as_list = [
        calculate_forecast(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            carry_price=carry_price,
            rule=rule,
        )
        for rule in rule_spec
    ]

    ### NOTE: This assumes we are equally weighted across spans
    ### eg all forecast weights the same, equally weighted
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    ## apply an FDM
    rule_count = len(rule_spec)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


FDM_LIST = {
    1: 1.0,
    2: 1.02,
    3: 1.03,
    4: 1.23,
    5: 1.25,
    6: 1.27,
    7: 1.29,
    8: 1.32,
    9: 1.34,
    10: 1.35,
    11: 1.36,
    12: 1.38,
    13: 1.39,
    14: 1.41,
    15: 1.42,
    16: 1.44,
    17: 1.46,
    18: 1.48,
    19: 1.50,
    20: 1.53,
    21: 1.54,
    22: 1.55,
    25: 1.69,
    30: 1.81,
    35: 1.93,
    40: 2.00,
}
fdm_x = list(FDM_LIST.keys())
fdm_y = list(FDM_LIST.values())

## We do this outside a function to avoid doing over and over
f_interp = interp1d(fdm_x, fdm_y, bounds_error=False, fill_value=2)


def get_fdm(rule_count):
    fdm = float(f_interp(rule_count))
    return fdm


def calculate_forecast(
    stdev_ann_perc: standardDeviation,
    carry_price: pd.DataFrame,
    adjusted_price: pd.Series,
    rule: dict,
) -> pd.Series:

    if rule["function"] == "carry":
        span = rule["span"]
        forecast = calculate_forecast_for_carry(
            stdev_ann_perc=stdev_ann_perc, carry_price=carry_price, span=span
        )

    elif rule["function"] == "ewmac":
        fast_span = rule["fast_span"]
        forecast = calculate_forecast_for_ewmac(
            adjusted_price=adjusted_price,
            stdev_ann_perc=stdev_ann_perc,
            fast_span=fast_span,
        )
    else:
        raise Exception("Rule %s not recognised!" % rule["function"])

    return forecast
    
def calculate_position_dict_with_forecast_from_function_applied(
    adjusted_prices_dict: dict,
    average_position_contracts_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> dict:


    list_of_instruments = list(adjusted_prices_dict.keys())

    # Create position dictionary by applying forecasts
    position_dict_with_carry = {
        instrument_code: calculate_position_with_forecast_applied_from_function(
            instrument_code=instrument_code,
            average_position_contracts_dict=average_position_contracts_dict,
            adjusted_prices_dict=adjusted_prices_dict,
            std_dev_dict=std_dev_dict,
            carry_prices_dict=carry_prices_dict,
            list_of_rules=list_of_rules,
        )
        for instrument_code in list_of_instruments
    }

    return position_dict_with_carry
def calculate_position_with_forecast_applied_from_function(
    instrument_code: str,
    average_position_contracts_dict: dict,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:


    # Calculate the combined forecast based on specified rules
    forecast = calculate_combined_forecast_from_function(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        list_of_rules=list_of_rules,
    )

    # Retrieve the average position for the instrument and scale by forecast
    average_position = average_position_contracts_dict[instrument_code]
    return forecast * average_position / 10
def calculate_combined_forecast_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:


    # Apply each rule (carry, ewmac, etc.) and store the results
    all_forecasts = [
        apply_forecast_function(
            instrument_code=instrument_code,
            adjusted_prices_dict=adjusted_prices_dict,
            std_dev_dict=std_dev_dict,
            carry_prices_dict=carry_prices_dict,
            rule=rule,
        )
        for rule in list_of_rules
    ]

    # Combine all forecasts (equally weighted)
    all_forecasts_df = pd.concat(all_forecasts, axis=1)
    average_forecast = all_forecasts_df.mean(axis=1)

    # Apply Forecast Diversification Multiplier (FDM)
    rule_count = len(list_of_rules)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm

    # Cap the forecast values between -20 and 20
    return scaled_forecast.clip(-20, 20)


def apply_forecast_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule: dict,
) -> pd.Series:


    # Check the function type in the rule and apply the corresponding forecast
    if rule["function"] == "carry":
        span = rule["span"]
        forecast = calculate_forecast_for_carry(
            stdev_ann_perc=std_dev_dict[instrument_code],
            carry_price=carry_prices_dict[instrument_code],
            span=span,
        )
    elif rule["function"] == "ewmac":
        fast_span = rule["fast_span"]
        forecast = calculate_forecast_for_ewmac(
            adjusted_price=adjusted_prices_dict[instrument_code],
            stdev_ann_perc=std_dev_dict[instrument_code],
            fast_span=fast_span,
        )
    else:
        raise Exception(f"Forecast function '{rule['function']}' not recognized!")

    return forecast
def calculate_combined_forecast_from_functions(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    list_of_rules: list,
) -> pd.Series:
 

    # Apply each forecasting rule in the list to calculate individual forecasts
    all_forecasts_as_list = [
        calculate_forecast_from_function(
            instrument_code=instrument_code,
            adjusted_prices_dict=adjusted_prices_dict,
            std_dev_dict=std_dev_dict,
            carry_prices_dict=carry_prices_dict,
            rule=rule,
        )
        for rule in list_of_rules
    ]

    # Combine the forecasts (equally weighted)
    all_forecasts_as_df = pd.concat(all_forecasts_as_list, axis=1)
    average_forecast = all_forecasts_as_df.mean(axis=1)

    # Apply the Forecast Diversification Multiplier (FDM)
    rule_count = len(list_of_rules)
    fdm = get_fdm(rule_count)
    scaled_forecast = average_forecast * fdm

    # Cap the forecast values between -20 and 20 to avoid extreme positions
    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast
def calculate_forecast_from_function(
    instrument_code: str,
    adjusted_prices_dict: dict,
    std_dev_dict: dict,
    carry_prices_dict: dict,
    rule: dict,
) -> pd.Series:

    
    # Extract the forecasting function and scalar from the rule
    rule_function = rule.pop("function")
    scalar = rule.pop("scalar")
    rule_args = rule

    # Apply the forecasting function and scale the result
    forecast_value = rule_function(
        instrument_code=instrument_code,
        adjusted_prices_dict=adjusted_prices_dict,
        std_dev_dict=std_dev_dict,
        carry_prices_dict=carry_prices_dict,
        **rule_args
    )

    return forecast_value * scalar



if __name__ == "__main__":
    
    instrument_list1 = ["SP500","US10"]
    (
        adjusted_prices_dict,
        current_prices_dict,
        carry_prices_dict,
    ) = get_data_dict_with_carry(instrument_list=instrument_list1)
    

    multipliers = dict(SP500=5, US10=10000)
    risk_target_tau = 0.25
    fx_series_dict = create_fx_series_given_adjusted_prices_dict(adjusted_prices_dict)

    capital = 20000000

    idm = 6
    instrument_weights = dict(SP500=0.5, US10=0.5)
    cost_per_contract_dict = dict(SP500=0.875, US10=15.3)

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

    ## Assumes equal forecast weights and we use all rules for both instruments
    rules_spec_ewmac = [
        dict(function="ewmac", fast_span=8),
        dict(function="ewmac", fast_span=16),
        dict(function="ewmac", fast_span=32),
    ]
    position_contracts_dict_ewmac = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec_ewmac,
    )

    buffered_position_dict_ewmac = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict_ewmac,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict_ewmac = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict_ewmac,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    perc_return_aggregated_ewmac = aggregate_returns(perc_return_dict_ewmac)

    rules_spec_carry = [
        dict(function="carry", span=5),
        dict(function="carry", span=20),
        dict(function="carry", span=60),
        dict(function="carry", span=120),
    ]
    position_contracts_dict_carry = calculate_position_dict_with_forecast_applied(
        adjusted_prices_dict=adjusted_prices_dict,
        carry_prices_dict=carry_prices_dict,
        std_dev_dict=std_dev_dict,
        average_position_contracts_dict=average_position_contracts_dict,
        rule_spec=rules_spec_carry,
    )

    buffered_position_dict_carry = apply_buffering_to_position_dict(
        position_contracts_dict=position_contracts_dict_carry,
        average_position_contracts_dict=average_position_contracts_dict,
    )

    perc_return_dict_carry = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=buffered_position_dict_carry,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    perc_return_aggregated_carry = aggregate_returns(perc_return_dict_carry)

    starting_portfolio = (
        perc_return_aggregated_ewmac * 0.6 + perc_return_aggregated_carry * 0.4
    )

    relative_performance = perc_return_aggregated_ewmac - perc_return_aggregated_carry
    rolling_12_month = relative_performance.rolling(256).mean()

    relative_performance = perc_return_aggregated_ewmac - perc_return_aggregated_carry
    rolling_12_month = (
        relative_performance.rolling(256).sum() / risk_target_tau
    )

    # W t = EWMA span=30 (min(1, max(0, 0.5 + RP t ÷ 2)))
    raw_weighting = 0.5 + rolling_12_month / 2
    clipped_weighting = raw_weighting.clip(lower=0, upper=1)
    smoothed_weighting = clipped_weighting.ewm(30).mean()

    weighted_portfolio = (
        perc_return_aggregated_ewmac * 0.6 * smoothed_weighting
        + perc_return_aggregated_carry * 0.4 * (1 - smoothed_weighting))
    weighted_portfolio_stats = calculate_stats(weighted_portfolio)
    ewmac_stats = calculate_stats(perc_return_aggregated_ewmac)
    carry_stats = calculate_stats(perc_return_aggregated_carry)

    print("Weighted Portfolio Stats:", weighted_portfolio_stats)
    print("EWMAC Stats:", ewmac_stats)
    print("Carry Stats:", carry_stats)