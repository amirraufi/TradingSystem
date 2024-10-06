import pandas as pd 
import copy



def calculate_perc_returns(position_contracts_held: pd.Series,
                            adjusted_price: pd.Series,
                           fx_series: pd.Series,
                           multiplier: float,
                           capital_required: pd.Series,
                           ) -> pd.Series:

    return_price_points = (adjusted_price - adjusted_price.shift(1))*position_contracts_held.shift(1)

    return_instrument_currency = return_price_points * multiplier
    fx_series_aligned = fx_series.reindex(return_instrument_currency.index, method="ffill")
    return_base_currency = return_instrument_currency * fx_series_aligned

    perc_return = return_base_currency / capital_required

    return perc_return


def calculate_standard_deviation_for_risk_targeting(adjusted_price: pd.Series,
                                                    current_price: pd.Series):

    daily_price_changes = adjusted_price.diff()
    percentage_changes = daily_price_changes / current_price.shift(1)

    ## Can do the whole series or recent history
    recent_daily_std = percentage_changes.tail(30).std()

    return recent_daily_std*(256**.5)
def calculate_position_series_given_fixed_risk(capital: float,
                                               risk_target_tau: float,
                                               current_price: pd.Series,
                                               fx: pd.Series,
                                               multiplier: float,
                                               instrument_risk_ann_perc: float) -> pd.Series:

    #N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    position_in_contracts =  capital * risk_target_tau / (multiplier * current_price * fx * instrument_risk_ann_perc)

    return position_in_contracts
def calculate_minimum_capital(multiplier: float,
                              price: float,
                              fx: float,
                              instrument_risk_ann_perc: float,
                              risk_target: float,
                              contracts: int = 4):
    # (4 × Multiplier × Price × FX × σ % ) ÷ τ
    minimum_capital= contracts * multiplier * price * fx * instrument_risk_ann_perc / risk_target
    return minimum_capital
def calculate_position_series_given_fixed_risk(capital, risk_target_tau, current_price, fx, multiplier, instrument_risk_ann_perc):
    return capital * risk_target_tau / (multiplier * current_price * fx * instrument_risk_ann_perc)

def calculate_percentage_returns(
    adjusted_price: pd.Series, current_price: pd.Series
) -> pd.Series:

    daily_price_changes = calculate_daily_returns(adjusted_price)
    percentage_changes = daily_price_changes / current_price.shift(1)

    return percentage_changes

def calculate_daily_returns(adjusted_price: pd.Series) -> pd.Series:

    return adjusted_price.diff()
def calculate_variable_standard_deviation_for_risk_targeting(
    adjusted_price: pd.Series,
    current_price: pd.Series,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> pd.Series:

    if use_perc_returns:
        daily_returns = calculate_percentage_returns(
            adjusted_price=adjusted_price, current_price=current_price
        )
    else:
        daily_returns = calculate_daily_returns(adjusted_price=adjusted_price)

    ## Can do the whole series or recent history
    daily_exp_std_dev = daily_returns.ewm(span=32).std()

    if annualise_stdev:
        annualisation_factor = 256 ** 0.5
    else:
        ## leave at daily
        annualisation_factor = 1

    annualised_std_dev = daily_exp_std_dev * annualisation_factor

    ## Weight with ten year vol
    ten_year_vol = annualised_std_dev.rolling(
        256 * 10, min_periods=1
    ).mean()
    weighted_vol = 0.3 * ten_year_vol + 0.7 * annualised_std_dev

    return weighted_vol


class standardDeviation(pd.Series):
    ## class that can be eithier % or price based standard deviation estimate
    def __init__(
        self,
        adjusted_price: pd.Series,
        current_price: pd.Series,
        use_perc_returns: bool = True,
        annualise_stdev: bool = True,
    ):

        stdev = calculate_variable_standard_deviation_for_risk_targeting(
            adjusted_price=adjusted_price,
            current_price=current_price,
            annualise_stdev=annualise_stdev,
            use_perc_returns=use_perc_returns,
        )
        super().__init__(stdev)

        self._use_perc_returns = use_perc_returns
        self._annualised = annualise_stdev
        self._current_price = current_price

    def daily_risk_price_terms(self):
        stdev = copy.copy(self)
        if self.annualised:
            stdev = stdev / (256 ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    def annual_risk_price_terms(self):
        stdev = copy.copy(self)
        if not self.annualised:
            # daily
            stdev = stdev * (256 ** 0.5)

        if self.use_perc_returns:
            stdev = stdev * self.current_price

        return stdev

    @property
    def annualised(self) -> bool:
        return self._annualised

    @property
    def use_perc_returns(self) -> bool:
        return self._use_perc_returns

    @property
    def current_price(self) -> pd.Series:
        return self._current_price

def calculate_position_series_given_variable_risk(
    capital: float,
    risk_target_tau: float,
    fx: pd.Series,
    multiplier: float,
    instrument_risk: standardDeviation,
) -> pd.Series:

    # N = (Capital × τ) ÷ (Multiplier × Price × FX × σ %)
    ## resolves to N = (Capital × τ) ÷ (Multiplier × FX × daily stdev price terms × 16)
    ## for simplicity we use the daily risk in price terms, even if we calculated annualised % returns
    daily_risk_price_terms = instrument_risk.daily_risk_price_terms()

    return (
        capital
        * risk_target_tau
        / (multiplier * fx * daily_risk_price_terms * (256 ** 0.5))
    )


def calculate_turnover(position, average_position):
    daily_trades = position.diff()
    as_proportion_of_average = daily_trades.abs() / average_position.shift(1)
    average_daily = as_proportion_of_average.mean()
    annualised_turnover = average_daily * 256

    return annualised_turnover

def calculate_minimum_capital(multiplier, price, fx, instrument_risk_ann_perc, risk_target, contracts=4):
    return contracts * multiplier * price * fx * instrument_risk_ann_perc / risk_target

def aggregate_returns(perc_returns_dict: dict) -> pd.Series:
    both_returns = perc_returns_to_df(perc_returns_dict)
    agg = both_returns.sum(axis=1)
    return agg
def perc_returns_to_df(perc_returns_dict: dict) -> pd.DataFrame:
    both_returns = pd.concat(perc_returns_dict, axis=1)
    both_returns = both_returns.dropna(how="all")

    return both_returns

def calculate_variable_standard_deviation_for_risk_targeting_from_dict(
    adjusted_prices: dict,
    current_prices: dict,
    use_perc_returns: bool = True,
    annualise_stdev: bool = True,
) -> dict:

    std_dev_dict = dict(
        [
            (
                instrument_code,
                standardDeviation(
                    adjusted_price=adjusted_prices[instrument_code],
                    current_price=current_prices[instrument_code],
                    use_perc_returns=use_perc_returns,
                    annualise_stdev=annualise_stdev,
                ),
            )
            for instrument_code in adjusted_prices.keys()
        ]
    )

    return std_dev_dict



def calculate_position_series_given_variable_risk_for_dict(
    capital: float,
    risk_target_tau: float,
    idm: float,
    weights: dict,
    fx_series_dict: dict,
    multipliers: dict,
    std_dev_dict: dict,
) -> dict:

    position_series_dict = dict(
        [
            (
                instrument_code,
                calculate_position_series_given_variable_risk(
                    capital=capital * idm * weights[instrument_code],
                    risk_target_tau=risk_target_tau,
                    multiplier=multipliers[instrument_code],
                    fx=fx_series_dict[instrument_code],
                    instrument_risk=std_dev_dict[instrument_code],
                ),
            )
            for instrument_code in std_dev_dict.keys()
        ]
    )

    return position_series_dict


def calculate_perc_returns_for_dict(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
) -> dict:

    perc_returns_dict = dict(
        [
            (
                instrument_code,
                calculate_perc_returns(
                    position_contracts_held=position_contracts_dict[instrument_code],
                    adjusted_price=adjusted_prices[instrument_code],
                    multiplier=multipliers[instrument_code],
                    fx_series=fx_series[instrument_code],
                    capital_required=capital,
                ),
            )
            for instrument_code in position_contracts_dict.keys()
        ]
    )

    return perc_returns_dict

def minimum_capital_for_sub_strategy(multiplier, price, fx, instrument_risk_ann_perc, risk_target, contracts=4):
    return contracts * multiplier * price * fx * instrument_risk_ann_perc / risk_target

def calculate_perc_returns_for_dict_with_costs(
    position_contracts_dict: dict,
    adjusted_prices: dict,
    multipliers: dict,
    fx_series: dict,
    capital: float,
    cost_per_contract_dict: dict,
    std_dev_dict: dict,
) -> dict:

    perc_returns_dict = dict(
        [
            (
                instrument_code,
                calculate_perc_returns_with_costs(
                    position_contracts_held=position_contracts_dict[instrument_code],
                    adjusted_price=adjusted_prices[instrument_code],
                    multiplier=multipliers[instrument_code],
                    fx_series=fx_series[instrument_code],
                    capital_required=capital,
                    cost_per_contract=cost_per_contract_dict[instrument_code],
                    stdev_series=std_dev_dict[instrument_code],
                ),
            )
            for instrument_code in position_contracts_dict.keys()
        ]
    )

    return perc_returns_dict


def calculate_perc_returns_with_costs(
    position_contracts_held: pd.Series,
    adjusted_price: pd.Series,
    fx_series: pd.Series,
    stdev_series: standardDeviation,
    multiplier: float,
    capital_required: float,
    cost_per_contract: float,
) -> pd.Series:

    precost_return_price_points = (
        adjusted_price - adjusted_price.shift(1)
    ) * position_contracts_held.shift(1)

    precost_return_instrument_currency = precost_return_price_points * multiplier
    historic_costs = calculate_costs_deflated_for_vol(
        stddev_series=stdev_series,
        cost_per_contract=cost_per_contract,
        position_contracts_held=position_contracts_held,
    )

    historic_costs_aligned = historic_costs.reindex(
        precost_return_instrument_currency.index, method="ffill"
    )
    return_instrument_currency = (
        precost_return_instrument_currency - historic_costs_aligned
    )

    fx_series_aligned = fx_series.reindex(
        return_instrument_currency.index, method="ffill"
    )
    return_base_currency = return_instrument_currency * fx_series_aligned

    perc_return = return_base_currency / capital_required

    return perc_return


def calculate_costs_deflated_for_vol(
    stddev_series: standardDeviation,
    cost_per_contract: float,
    position_contracts_held: pd.Series,
) -> pd.Series:

    round_position_contracts_held = position_contracts_held.round()
    position_change = (
        round_position_contracts_held - round_position_contracts_held.shift(1)
    )
    abs_trades = position_change.abs()

    historic_cost_per_contract = calculate_deflated_costs(
        stddev_series=stddev_series, cost_per_contract=cost_per_contract
    )

    historic_cost_per_contract_aligned = historic_cost_per_contract.reindex(
        abs_trades.index, method="ffill"
    )

    historic_costs = abs_trades * historic_cost_per_contract_aligned

    return historic_costs


def calculate_deflated_costs(
    stddev_series: standardDeviation, cost_per_contract: float
) -> pd.Series:

    stdev_daily_price = stddev_series.daily_risk_price_terms()

    final_stdev = stdev_daily_price.iloc[-1]

    cost_deflator = stdev_daily_price / final_stdev
    historic_cost_per_contract = cost_per_contract * cost_deflator

    return historic_cost_per_contract


def long_only_returns(adjusted_prices_dict, std_dev_dict, average_position_contracts_dict, fx_series_dict, cost_per_contract_dict, multipliers, capital):
    perc_return_dict = calculate_perc_returns_for_dict_with_costs(
        position_contracts_dict=average_position_contracts_dict,
        fx_series=fx_series_dict,
        multipliers=multipliers,
        capital=capital,
        adjusted_prices=adjusted_prices_dict,
        cost_per_contract_dict=cost_per_contract_dict,
        std_dev_dict=std_dev_dict,
    )

    perc_return_agg = aggregate_returns(perc_return_dict)
    return perc_return_agg
def calculate_rolling_sharpe_ratio(
    relative_price_dict: dict, rolling_window: int = 256
) -> dict:
    rolling_sharpe_dict = {
        instrument: (relative_price.rolling(rolling_window).mean() / relative_price.rolling(rolling_window).std())
        for instrument, relative_price in relative_price_dict.items()
    }
    return rolling_sharpe_dict
