# import pandas as pd 
# import numpy as np 
# import yfinance as yf
# import matplotlib.pyplot as plt
# from sklearn.utils import resample
# from scipy.stats import skew
import datetime

# def annualised_sharper_ratio(ret,std):
#     16*(ret/std)

# tickers = {
#     "Gold" : "GC=F",
#     "Treasury" : "ZN=F",
#     "GBPUSD" : "GBPUSD=X",
#     "S&P500" :"ES=F"
# }
# def download_data(ticker, start_date = "2004-01-01", end_date="2024-09-18"):
#     data = yf.download(ticker, start= start_date, end= end_date)
#     data.dropna(inplace=True)
#     return data 
# data = {name : download_data(ticker) for name, ticker in tickers.items()}
# # print(data["S&P500"].head())
# # fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

# # # # # Plot each asset in its own subplot
# # for ax, (name, df) in zip(axes, data.items()):
# #     ax.plot(df.index, df['Close'], label=name)
# #     ax.set_title(f'{name} Futures')
# #     ax.set_xlabel('Date')
# #     ax.set_ylabel('Price')
# #     ax.legend(loc='upper left')
# #     ax.grid(True)

# # # # # Adjust layout to prevent overlap
# # plt.tight_layout()

# # # # Show the plots
# # plt.show()
# # # as the data from yahoo finance is already continuos we can not use the panama methode here so we will work with this data but if we had multiple future expirations we had to use panama to make the continuous.
# def ewma(data, window):
#     return data.ewm(span=window, adjust=False).mean()
# gold = data["Gold"]
# # Short, Medium, and Long Term EWMA
# def exponential_momentum_generator(x):
#     x['EWMA_2'] = ewma(x['Close'], 2)
#     x['EWMA_4'] = ewma(x['Close'], 4)
#     x['EWMA_8'] = ewma(x['Close'], 8)
#     x['EWMA_16'] = ewma(x['Close'], 16)
#     x['EWMA_32'] = ewma(x['Close'], 32)
#     x['EWMA_64'] = ewma(x['Close'], 64)
#     x['EWMA_128'] = ewma(x['Close'], 128)
#     x['EWMA_256'] = ewma(x["Close"], 256)
# exponential_momentum_generator(gold)
# # Plot the data
# # plt.figure(figsize=(12, 6))
# # plt.plot(gold.index, gold['Close'], label='Gold Close Price', color='black')
# # plt.plot(gold.index, gold['EWMA_8'], label='EWMA 8 (Very Short Term)', color='blue')
# # plt.plot(gold.index, gold['EWMA_16'], label='EWMA 16 (Short Term)', color='green')
# # plt.plot(gold.index, gold['EWMA_32'], label='EWMA 32 (Medium Term)', color='yellow')
# # plt.plot(gold.index, gold['EWMA_64'], label='EWMA 64 (Long Term)', color='orange')
# # plt.plot(gold.index, gold['EWMA_128'], label='EWMA 128 (Very Long Term)', color='red')
# def EWMA_Signal(x):
#     x['Signal_2_8']=np.where(x['EWMA_2'] > x['EWMA_8'], 1, -1)
#     x['Signal_4_16']=np.where(x['EWMA_4'] > x['EWMA_16'], 1, -1)
#     x['Signal_8_32']=np.where(x['EWMA_8'] > x['EWMA_32'], 1, -1)
#     x['Signal_16_64']=np.where(x['EWMA_16'] > x['EWMA_64'], 1, -1)
#     x['Signal_32_128']=np.where(x['EWMA_32'] > x['EWMA_128'],1,-1)
#     x['Signal_64_256']=np.where(x['EWMA_64'] > x['EWMA_256'],1,-1)
# # gold['Signal_8_16'] = np.where(gold['EWMA_8'] > gold['EWMA_16'], 1, -1)  # Buy when 8 crosses above 16, Sell when crosses below
# # gold['Signal_16_32'] = np.where(gold['EWMA_16'] > gold['EWMA_32'], 1, -1)  # Buy when 16 crosses above 32, Sell when crosses below
# # gold['Signal_32_64'] = np.where(gold['EWMA_32'] > gold['EWMA_64'], 1, -1)  # Buy when 32 crosses above 64, Sell when crosses below
# # gold['Signal_64_128'] = np.where(gold['EWMA_64'] > gold['EWMA_128'], 1, -1) # Buy when 64 crosses above 128, Sell when crosses below
# # Calculate the correlation between the signals
# EWMA_Signal(gold)
# signals_df = gold[['Signal_2_8', 'Signal_4_16', 'Signal_8_32',"Signal_16_64", 'Signal_32_128','Signal_64_256']].dropna()
# signal_correlation = signals_df.corr()

# # Print the correlation matrix
# print("Correlation between different EWMA signals:")
# print(signal_correlation)
# # plt.title('Gold Futures with Exponentially Weighted Moving Averages')
# # plt.xlabel('Date')
# # plt.ylabel('Price')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# def bootstrap_correlation(data, n_iterations=1000, subset_size=0.1):
#     correlation_matrices = []

#     # Perform bootstrapping
#     for i in range(n_iterations):
#         # Randomly sample a subset of the data (with replacement)
#         bootstrap_sample = resample(data, n_samples=int(len(data) * subset_size))
        
#         # Calculate correlation matrix for the subset
#         corr_matrix = bootstrap_sample.corr()
#         correlation_matrices.append(corr_matrix)

#     # Calculate the mean correlation matrix across all bootstrap samples
#     avg_correlation = np.mean(correlation_matrices, axis=0)
    
#     return avg_correlation

# # Calculate the out-of-sample bootstrapped correlation
# bootstrap_corr = bootstrap_correlation(signals_df, n_iterations=100, subset_size=0.1)

# print("Bootstrapped Correlation Matrix:")
# print(bootstrap_corr)
# gold["returns"] = np.log(gold["Close"] / gold["Close"].shift(1))

# # Initialize position: Start at 0, adjust based on signals
# gold['Position'] = 0

# # When the signal is positive, hold the position (buy), when it's negative, reduce or sell
# gold['Position'] = np.where(gold["Signal_64_256"] > 0, 1, np.where(gold["Signal_64_256"] < 0, -1, 0))

# # Shift position to reflect that today's position influences tomorrow's return
# gold['Position'] = gold['Position'].shift(1)

# # Calculate strategy returns based on the position held
# gold["Strategy_returns"] = gold['Position'] * gold['returns']

# # Drop NaN values
# gold.dropna(inplace=True)

# # Calculate skewness of the strategy's returns
# strategy_skewness = skew(gold["Strategy_returns"])
# print(f"Skewness of strategy returns: {strategy_skewness}")
# strategy_std = gold["Strategy_returns"].std()
# print(f"Standard Deviation of strategy log returns: {strategy_std}")

# # Calculate cumulative log return by summing the strategy's log returns
# strategy_cumulative_return = gold['Strategy_returns'].sum()

# # Convert cumulative log return to percentage terms
# strategy_total_return_percentage = np.exp(strategy_cumulative_return) - 1
# print(f"Cumulative Return of strategy: {strategy_total_return_percentage * 100:.2f}%")
# print(gold)
