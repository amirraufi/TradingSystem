import pandas as pd
import numpy as np


def apply_buffering_to_position_dict(
    position_contracts_dict: dict, average_position_contracts_dict: dict
) -> dict:

    instrument_list = list(position_contracts_dict.keys())
    buffered_position_dict = dict(
        [
            (
                instrument_code,
                apply_buffering_to_positions(
                    position_contracts=position_contracts_dict[instrument_code],
                    average_position_contracts=average_position_contracts_dict[
                        instrument_code
                    ],
                ),
            )
            for instrument_code in instrument_list
        ]
    )

    return buffered_position_dict


def apply_buffering_to_positions(
    position_contracts: pd.Series,
    average_position_contracts: pd.Series,
    buffer_size: float = 0.10,
) -> pd.Series:

    buffer = average_position_contracts.abs() * buffer_size
    upper_buffer = position_contracts + buffer
    lower_buffer = position_contracts - buffer

    buffered_position = apply_buffer(
        optimal_position=position_contracts,
        upper_buffer=upper_buffer,
        lower_buffer=lower_buffer,
    )

    return buffered_position


def apply_buffer(
    optimal_position: pd.Series, upper_buffer: pd.Series, lower_buffer: pd.Series
) -> pd.Series:

    upper_buffer = upper_buffer.ffill().round()
    lower_buffer = lower_buffer.ffill().round()
    use_optimal_position = optimal_position.ffill()

    current_position = use_optimal_position[0]
    if np.isnan(current_position):
        current_position = 0.0

    buffered_position_list = [current_position]

    for idx in range(len(optimal_position.index))[1:]:
        current_position = apply_buffer_single_period(
            last_position=current_position,
            top_pos=upper_buffer[idx],
            bot_pos=lower_buffer[idx],
        )

        buffered_position_list.append(current_position)

    buffered_position = pd.Series(buffered_position_list, index=optimal_position.index)

    return buffered_position


def apply_buffer_single_period(last_position: int, top_pos: float, bot_pos: float):

    if last_position > top_pos:
        return top_pos
    elif last_position < bot_pos:
        return bot_pos
    else:
        return last_position