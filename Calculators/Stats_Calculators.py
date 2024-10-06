from datetime import date
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
def pd_readcsv(
    filename: str,
        date_format=DEFAULT_DATE_FORMAT,
        date_index_name: str="index",
) -> pd.DataFrame:

    ans = pd.read_csv(filename)
    ans.index = pd.to_datetime(ans[date_index_name], format=date_format).values

    del ans[date_index_name]

    ans.index.name = None

    return ans
Frequency = Enum("Frequency", "Natural Year Month Week BDay")
NATURAL = Frequency.Natural
YEAR = Frequency.Year
MONTH = Frequency.Month
WEEK = Frequency.Week

def periods_per_year(at_frequency: Frequency):
    if at_frequency == NATURAL:
        return BUSINESS_DAYS_IN_YEAR
    else:
        return PERIODS_PER_YEAR[at_frequency]



def years_in_data(some_data: pd.Series) -> float:
    datediff = some_data.index[-1] - some_data.index[0]
    seconds_in_data = datediff.total_seconds()
    return seconds_in_data / SECONDS_PER_YEAR


def sum_at_frequency(perc_return: pd.Series,
                     at_frequency: Frequency = NATURAL) -> pd.Series:

    if at_frequency == NATURAL:
        return perc_return

    at_frequency_str_dict = {
                        YEAR: "Y",
                        WEEK: "7D",
                        MONTH: "1M"}
    at_frequency_str = at_frequency_str_dict[at_frequency]

    perc_return_at_freq = perc_return.resample(at_frequency_str).sum()

    return perc_return_at_freq


def ann_mean_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:

    mean_at_frequency = perc_return_at_freq.mean()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_mean = mean_at_frequency * periods_per_year_for_frequency

    return annualised_mean

def ann_std_given_frequency(perc_return_at_freq: pd.Series,
                             at_frequency: Frequency) -> float:

    std_at_frequency = perc_return_at_freq.std()
    periods_per_year_for_frequency = periods_per_year(at_frequency)
    annualised_std = std_at_frequency * (periods_per_year_for_frequency**.5)

    return annualised_std




def calculate_drawdown(perc_return):
    cum_perc_return = perc_return.cumsum()
    max_cum_perc_return = cum_perc_return.rolling(len(perc_return)+1,
                                                  min_periods=1).max()
    return max_cum_perc_return - cum_perc_return

QUANT_PERCENTILE_EXTREME = 0.01
QUANT_PERCENTILE_STD = 0.3
NORMAL_DISTR_RATIO = norm.ppf(QUANT_PERCENTILE_EXTREME) / norm.ppf(QUANT_PERCENTILE_STD)

def calculate_quant_ratio_lower(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def calculate_quant_ratio_upper(x):
    x_dm = demeaned_remove_zeros(x)
    raw_ratio = x_dm.quantile(1 - QUANT_PERCENTILE_EXTREME) / x_dm.quantile(
        1 - QUANT_PERCENTILE_STD
    )
    return raw_ratio / NORMAL_DISTR_RATIO

def demeaned_remove_zeros(x):
    x[x == 0] = np.nan
    return x - x.mean()
def calculate_stats(perc_return: pd.Series,
                at_frequency: Frequency = NATURAL) -> dict:

    perc_return_at_freq = sum_at_frequency(perc_return, at_frequency=at_frequency)

    ann_mean = ann_mean_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    ann_std = ann_std_given_frequency(perc_return_at_freq, at_frequency=at_frequency)
    sharpe_ratio = ann_mean / ann_std

    skew_at_freq = perc_return_at_freq.skew()
    drawdowns = calculate_drawdown(perc_return_at_freq)
    avg_drawdown = drawdowns.mean()
    max_drawdown = drawdowns.max()
    quant_ratio_lower = calculate_quant_ratio_upper(perc_return_at_freq)
    quant_ratio_upper = calculate_quant_ratio_upper(perc_return_at_freq)

    return dict(
        ann_mean = ann_mean,
        ann_std = ann_std,
        sharpe_ratio = sharpe_ratio,
        skew = skew_at_freq,
        avg_drawdown = avg_drawdown,
        max_drawdown = max_drawdown,
        quant_ratio_lower = quant_ratio_lower,
        quant_ratio_upper = quant_ratio_upper
    )

BUSINESS_DAYS_IN_YEAR = 256
WEEKS_PER_YEAR = 52.25
MONTHS_PER_YEAR = 12
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

PERIODS_PER_YEAR = {
    MONTH: MONTHS_PER_YEAR,
    WEEK: WEEKS_PER_YEAR,
    YEAR: 1

}