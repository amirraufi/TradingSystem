import pandas as pd
import numpy as np
from dataclasses import dataclass

def generate_pandl_across_instruments_for_hourly_data(
    adjusted_prices_daily_dict: dict,
    current_prices_daily_dict: dict,
    adjusted_prices_hourly_dict: dict,
    std_dev_dict: dict,
    average_position_contracts_dict: dict,
    fx_series_dict: dict,
    multipliers: dict,
    commission_per_contract_dict: dict,
    capital: float,
    tick_size_dict: dict,
    bid_ask_spread_dict: dict,
    trade_calculation_function,
) -> dict:

    list_of_instruments = list(adjusted_prices_hourly_dict.keys())

    pandl_dict = dict(
        [
            (
                instrument_code,
                calculate_pandl_series_for_instrument(
                    adjusted_hourly_prices=adjusted_prices_hourly_dict[instrument_code],
                    current_daily_prices=current_prices_daily_dict[instrument_code],
                    adjusted_daily_prices=adjusted_prices_daily_dict[instrument_code],
                    daily_stdev=std_dev_dict[instrument_code],
                    average_position_daily=average_position_contracts_dict[
                        instrument_code
                    ],
                    fx_series=fx_series_dict[instrument_code],
                    multiplier=multipliers[instrument_code],
                    commission_per_contract=commission_per_contract_dict[
                        instrument_code
                    ],
                    tick_size=tick_size_dict[instrument_code],
                    bid_ask_spread=bid_ask_spread_dict[instrument_code],
                    capital=capital,
                    trade_calculation_function=trade_calculation_function,
                    instrument_code=instrument_code,
                ),
            )
            for instrument_code in list_of_instruments
        ]
    )

    return pandl_dict


def calculate_pandl_series_for_instrument(
    adjusted_daily_prices: pd.Series,
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
    average_position_daily: pd.Series,
    fx_series: pd.Series,
    multiplier: float,
    capital: float,
    tick_size: float,
    bid_ask_spread: float,
    commission_per_contract: float,
    instrument_code: str,
    trade_calculation_function,
) -> pd.Series:

    list_of_trades = generate_list_of_mr_trades_for_instrument(
        adjusted_daily_prices=adjusted_daily_prices,
        current_daily_prices=current_daily_prices,
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_stdev=daily_stdev,
        average_position_daily=average_position_daily,
        tick_size=tick_size,
        bid_ask_spread=bid_ask_spread,
        trade_calculation_function=trade_calculation_function,
        instrument_code=instrument_code,
    )

    perc_returns = calculate_perc_returns_from_trade_list(
        list_of_trades=list_of_trades,
        capital=capital,
        fx_series=fx_series,
        commission_per_contract=commission_per_contract,
        current_price_series=current_daily_prices,
        multiplier=multiplier,
        daily_stdev=daily_stdev,
    )

    return perc_returns


def generate_list_of_mr_trades_for_instrument(
    adjusted_daily_prices: pd.Series,
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
    average_position_daily: pd.Series,
    tick_size: float,
    bid_ask_spread: float,
    instrument_code: str,
    trade_calculation_function,
):

    daily_equilibrium_hourly = calculate_equilibrium(
        adjusted_hourly_prices=adjusted_hourly_prices,
        adjusted_daily_prices=adjusted_daily_prices,
    )

    hourly_stdev_prices = calculate_sigma_p(
        current_daily_prices=current_daily_prices,
        daily_stdev=daily_stdev,
        adjusted_hourly_prices=adjusted_hourly_prices,
    )

    list_of_trades = calculate_trades_for_instrument(
        adjusted_hourly_prices=adjusted_hourly_prices,
        daily_equilibrium_hourly=daily_equilibrium_hourly,
        tick_size=tick_size,
        bid_ask_spread=bid_ask_spread,
        hourly_stdev_prices=hourly_stdev_prices,
        average_position_daily=average_position_daily,
        trade_calculation_function=trade_calculation_function,
        instrument_code=instrument_code,
    )

    return list_of_trades


OrderType = Enum("OrderType", ["LIMIT", "MARKET"])


@dataclass
class Order:
    order_type: OrderType
    qty: int
    limit_price: float = np.nan

    @property
    def is_buy(self):
        return self.qty > 0

    @property
    def is_sell(self):
        return self.qty < 0


class ListOfOrders(list):
    def __init__(self, list_of_orders: List[Order]):
        super().__init__(list_of_orders)

    def drop_buy_limits(self):
        return self.drop_signed_limit_orders(1)

    def drop_sell_limits(self):
        return self.drop_signed_limit_orders(-1)

    def drop_signed_limit_orders(self, order_sign: int):
        new_list = ListOfOrders(
            [
                order
                for order in self
                if true_if_order_is_market_or_order_is_not_of_sign(order, order_sign)
            ]
        )
        return new_list


def true_if_order_is_market_or_order_is_not_of_sign(
    order: Order, order_sign_to_drop: int
):
    if order.order_type == OrderType.MARKET:
        return True

    if not np.sign(order.qty) == order_sign_to_drop:
        return True

    return False


def calculate_trades_for_instrument(
    adjusted_hourly_prices: pd.Series,
    daily_equilibrium_hourly: pd.Series,
    average_position_daily: pd.Series,
    hourly_stdev_prices: pd.Series,
    bid_ask_spread: float,
    tick_size: float,
    instrument_code: str,
    trade_calculation_function,
) -> list:

    list_of_trades = []
    list_of_orders_for_period = ListOfOrders([])
    current_position = 0
    list_of_dates = list(adjusted_hourly_prices.index)

    for relevant_date in list_of_dates[1:]:
        current_price = float(
            get_row_of_series_before_date(
                adjusted_hourly_prices, relevant_date=relevant_date
            )
        )
        if np.isnan(current_price):
            continue

        trade = fill_list_of_orders(
            list_of_orders_for_period,
            current_price=current_price,
            fill_date=relevant_date,
            bid_ask_spread=bid_ask_spread,
        )
        if trade.filled:
            list_of_trades.append(trade)
            current_position = current_position + trade.qty

        current_equilibrium = get_row_of_series_before_date(
            daily_equilibrium_hourly, relevant_date=relevant_date
        )
        current_average_position = get_row_of_series_before_date(
            average_position_daily, relevant_date=relevant_date
        )
        current_hourly_stdev_price = get_row_of_series_before_date(
            hourly_stdev_prices, relevant_date=relevant_date
        )

        list_of_orders_for_period = trade_calculation_function(
            current_position=current_position,
            current_price=current_price,
            current_equilibrium=current_equilibrium,
            current_average_position=current_average_position,
            current_hourly_stdev_price=current_hourly_stdev_price,
            tick_size=tick_size,
            instrument_code=instrument_code,
            relevant_date=relevant_date,
        )

    return list_of_trades


def required_orders_for_mr_system(
    current_position: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_price: float,
    current_average_position: float,
    tick_size: float,
    instrument_code: str,
    relevant_date,
) -> ListOfOrders:

    current_forecast = mr_forecast_unclipped(
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_price=current_price,
    )

    list_of_orders_for_period = calculate_orders_given_forecast_and_positions(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_position=current_position,
        tick_size=tick_size,
    )

    if current_forecast < -20:
        list_of_orders_for_period = list_of_orders_for_period.drop_sell_limits()
    elif current_forecast > 20:
        list_of_orders_for_period = list_of_orders_for_period.drop_buy_limits()

    return list_of_orders_for_period


def calculate_orders_given_forecast_and_positions(
    current_forecast: float,
    current_position: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
    tick_size: float,
) -> ListOfOrders:

    current_optimal_position = optimal_position_given_unclipped_forecast(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
    )

    trade_to_optimal = int(np.round(current_optimal_position - current_position))

    if abs(trade_to_optimal) > 1:
        list_of_orders = ListOfOrders(
            [Order(order_type=OrderType.MARKET, qty=trade_to_optimal)]
        )
        return list_of_orders

    buy_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        number_of_contracts_to_solve_for=current_position + 1,
    )

    sell_limit = get_limit_price_given_resulting_position_with_tick_size_applied(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        number_of_contracts_to_solve_for=current_position - 1,
    )

    return ListOfOrders(
        [
            Order(order_type=OrderType.LIMIT, qty=1, limit_price=buy_limit),
            Order(order_type=OrderType.LIMIT, qty=-1, limit_price=sell_limit),
        ]
    )


def mr_forecast_unclipped(
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_price: float,
) -> float:

    raw_forecast = current_equilibrium - current_price
    risk_adjusted_forecast = raw_forecast / current_hourly_stdev_price
    scaled_forecast = risk_adjusted_forecast * FORECAST_SCALAR

    return scaled_forecast


def optimal_position_given_unclipped_forecast(
    current_forecast: float, current_average_position: float
) -> float:

    clipped_forecast = np.clip(current_forecast, -20, 20)

    return clipped_forecast * current_average_position / AVG_ABS_FORECAST


def get_limit_price_given_resulting_position_with_tick_size_applied(
    number_of_contracts_to_solve_for: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
    tick_size: float,
) -> float:

    limit_price = get_limit_price_given_resulting_position(
        number_of_contracts_to_solve_for=number_of_contracts_to_solve_for,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
    )

    return np.round(limit_price / tick_size) * tick_size


def get_limit_price_given_resulting_position(
    number_of_contracts_to_solve_for: int,
    current_equilibrium: float,
    current_hourly_stdev_price: float,
    current_average_position: float,
) -> float:

    return current_equilibrium - (
        number_of_contracts_to_solve_for
        * AVG_ABS_FORECAST
        * current_hourly_stdev_price
        / (FORECAST_SCALAR * current_average_position)
    )


def generate_mr_forecast_series_for_instrument(
    daily_equilibrium_hourly: pd.Series,
    adjusted_hourly_prices: pd.Series,
    hourly_stdev_prices: pd.Series,
) -> pd.Series:

    adjusted_hourly_prices = adjusted_hourly_prices.squeeze()
    raw_forecast = daily_equilibrium_hourly - adjusted_hourly_prices
    risk_adjusted_forecast = raw_forecast / hourly_stdev_prices
    scaled_forecast = risk_adjusted_forecast * FORECAST_SCALAR

    capped_forecast = scaled_forecast.clip(-20, 20)

    return capped_forecast


def calculate_equilibrium(
    adjusted_daily_prices: pd.Series, adjusted_hourly_prices: pd.Series
):

    daily_equilibrium = adjusted_daily_prices.ewm(5).mean()
    daily_equilibrium_hourly = daily_equilibrium.reindex(
        adjusted_hourly_prices.index, method="ffill"
    )

    return daily_equilibrium_hourly


def calculate_sigma_p(
    current_daily_prices: pd.Series,
    adjusted_hourly_prices: pd.Series,
    daily_stdev: pd.Series,
):

    daily_stdev_prices = daily_stdev * current_daily_prices / 16
    hourly_stdev_prices = daily_stdev_prices.reindex(
        adjusted_hourly_prices.index, method="ffill"
    )

    return hourly_stdev_prices


import datetime


@dataclass
class Trade:
    qty: int
    fill_date: datetime.datetime
    current_price: float = np.nan

    @property
    def filled(self):
        return not self.unfilled

    @property
    def unfilled(self):
        return self.qty == 0


not_filled = object()


def fill_list_of_orders(
    list_of_orders: ListOfOrders,
    fill_date: datetime.datetime,
    current_price: float,
    bid_ask_spread: float,
) -> Trade:

    list_of_trades = [
        fill_order(
            order,
            current_price=current_price,
            fill_date=fill_date,
            bid_ask_spread=bid_ask_spread,
        )
        for order in list_of_orders
    ]
    list_of_trades = [trade for trade in list_of_trades if trade.filled]
    if len(list_of_trades) == 0:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)
    if len(list_of_trades) == 1:
        return list_of_trades[0]

    raise Exception("Impossible for multiple trades to be filled at a given level!")


def fill_order(
    order: Order,
    current_price: float,
    fill_date: datetime.datetime,
    bid_ask_spread: float,
) -> Trade:

    if order.order_type == OrderType.MARKET:
        return fill_market_order(
            order=order,
            current_price=current_price,
            fill_date=fill_date,
            bid_ask_spread=bid_ask_spread,
        )

    elif order.order_type == OrderType.LIMIT:
        return fill_limit_order(
            order=order, fill_date=fill_date, current_price=current_price
        )

    raise Exception("Order type not recognised!")


def fill_market_order(
    order: Order,
    current_price: float,
    fill_date: datetime.datetime,
    bid_ask_spread: float,
) -> Trade:

    if order.is_buy:
        return Trade(
            qty=order.qty,
            fill_date=fill_date,
            current_price=current_price + bid_ask_spread,
        )
    elif order.is_sell:
        return Trade(
            qty=order.qty,
            fill_date=fill_date,
            current_price=current_price - bid_ask_spread,
        )
    else:
        return Trade(qty=0, fill_date=fill_date, current_price=current_price)


def fill_limit_order(
    order: Order, fill_date: datetime.datetime, current_price: float
) -> Trade:

    if order.is_buy:
        if current_price > order.limit_price:
            return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    if order.is_sell:
        if current_price < order.limit_price:
            return Trade(qty=0, fill_date=fill_date, current_price=current_price)

    return Trade(current_price=order.limit_price, qty=order.qty, fill_date=fill_date)


def calculate_perc_returns_from_trade_list(
    list_of_trades: list,
    multiplier: float,
    capital: float,
    fx_series: pd.Series,
    current_price_series: pd.Series,
    commission_per_contract: float,
    daily_stdev: standardDeviation,
) -> pd.Series:

    trade_qty_as_list = [trade.qty for trade in list_of_trades]
    date_index_as_list = [trade.fill_date for trade in list_of_trades]
    price_index_as_list = [trade.current_price for trade in list_of_trades]

    trade_qty_as_series = pd.Series(trade_qty_as_list, index=date_index_as_list)
    trade_prices_as_series = pd.Series(price_index_as_list, index=date_index_as_list)
    position_series = trade_qty_as_series.cumsum()

    perc_returns = calculate_perc_returns_with_costs(
        position_contracts_held=position_series,
        adjusted_price=trade_prices_as_series,
        fx_series=fx_series,
        capital_required=capital,
        multiplier=multiplier,
        cost_per_contract=commission_per_contract,
        stdev_series=daily_stdev,
    )

    return perc_returns

from correlation_estimate import get_row_of_series_before_date

def build_ewmac_storage_dict(adjusted_price_dict: dict,
                             current_prices_daily_dict: dict,
                             std_dev_dict: dict) -> dict:
    list_of_instruments = list(adjusted_price_dict.keys())
    ewmac_dict = dict([
                        (
                           instrument_code,
                            calculate_forecast_for_ewmac(
                                adjusted_price=adjusted_price_dict[instrument_code],
                                 current_price=current_prices_daily_dict[instrument_code],
                                 stdev_ann_perc=std_dev_dict[instrument_code],
                                fast_span=16
                                 )
                        )
                    for instrument_code in list_of_instruments
        ])

    return ewmac_dict

def build_vol_attenuation_dict(
                             std_dev_dict: dict) -> dict:

    list_of_instruments = list(std_dev_dict.keys())
    ewmac_dict = dict([
                        (
                           instrument_code,
                            get_attenuation(std_dev_dict[instrument_code])
                        )
                    for instrument_code in list_of_instruments
        ])

    return ewmac_dict


def required_orders_for_mr_system_with_overlays(current_position: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_price: float,
                    current_average_position: float,
                    tick_size: float,
                    instrument_code: str,
                    relevant_date
                    ) -> ListOfOrders:

    current_forecast = mr_forecast_unclipped(current_equilibrium=current_equilibrium,
                                   current_hourly_stdev_price= current_hourly_stdev_price,
                                   current_price=current_price)

    ewmac_sign = current_ewmac_sign(instrument_code, relevant_date)
    if not np.sign(current_forecast)==ewmac_sign:
        current_forecast = 0

    current_atten = current_vol_atten(instrument_code, relevant_date)
    current_forecast = current_forecast * current_atten

    list_of_orders_for_period = calculate_orders_given_forecast_and_positions_and_overlay(
        current_average_position=current_average_position,
        current_forecast=current_forecast,
        current_equilibrium=current_equilibrium,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_position = current_position,
        tick_size=tick_size,
        current_atten = current_atten,
        ewmac_sign = ewmac_sign
    )

    if current_forecast<-20:
        list_of_orders_for_period = list_of_orders_for_period.drop_sell_limits()
    elif current_forecast>20:
        list_of_orders_for_period = list_of_orders_for_period.drop_buy_limits()

    return list_of_orders_for_period

def calculate_orders_given_forecast_and_positions_and_overlay(
        current_forecast: float,
        current_position: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        tick_size: float,
        ewmac_sign: float,
        current_atten: float
) -> ListOfOrders:

    if not current_position==0:
        if not ewmac_sign==np.sign(current_position):
            list_of_orders = ListOfOrders(
                [
                    Order(order_type=OrderType.MARKET,
                          qty=-current_position)
                ]
            )
            return list_of_orders

    current_optimal_position = optimal_position_given_unclipped_forecast(current_average_position=current_average_position,
                                    current_forecast=current_forecast)

    trade_to_optimal = int(np.round(current_optimal_position - current_position))

    if abs(trade_to_optimal)>1:
        list_of_orders = ListOfOrders(
            [
                Order(order_type=OrderType.MARKET,
                      qty=trade_to_optimal)
            ]
        )
        return list_of_orders


    buy_limit = get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        current_atten=current_atten,

        number_of_contracts_to_solve_for = current_position + 1,

    )

    sell_limit = get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_equilibrium=current_equilibrium,
        tick_size=tick_size,
        current_atten = current_atten,

        number_of_contracts_to_solve_for=current_position-1
    )

    return ListOfOrders([
        Order(order_type=OrderType.LIMIT,
              qty=1,
              limit_price=buy_limit),
        Order(order_type=OrderType.LIMIT,
              qty= -1,
              limit_price=sell_limit)
    ])



def current_ewmac_sign(instrument_code: str,
                       relevant_date) -> float:

    return np.sign(get_row_of_series_before_date(ewmac_dict[instrument_code],
                                         relevant_date))


def current_vol_atten(instrument_code: str,
                       relevant_date) -> float:
    return get_row_of_series_before_date(atten_dict[instrument_code],
                                                 relevant_date)

def get_limit_price_given_resulting_position_with_tick_size_applied_for_overlay(
        number_of_contracts_to_solve_for: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        tick_size: float,
        current_atten: float

)-> float:

    limit_price = \
        get_limit_price_given_resulting_position_with_overlay(
            number_of_contracts_to_solve_for= number_of_contracts_to_solve_for,
        current_equilibrium=current_equilibrium,
        current_average_position=current_average_position,
        current_hourly_stdev_price=current_hourly_stdev_price,
        current_atten=current_atten)

    return np.round(limit_price / tick_size) * tick_size

def get_limit_price_given_resulting_position_with_overlay(
        number_of_contracts_to_solve_for: int,
        current_equilibrium: float,
        current_hourly_stdev_price: float,
        current_average_position: float,
        current_atten: float

)-> float:

    return current_equilibrium - (number_of_contracts_to_solve_for *
                                  AVG_ABS_FORECAST *
                                  current_hourly_stdev_price /
                                  (FORECAST_SCALAR * current_atten * current_average_position))

