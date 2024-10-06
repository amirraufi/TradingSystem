import pandas as pd 
# file_path = ""
# df = pd.read_csv(file_path, parse_dates=['DATETIME'])

# df['DATETIME'] = pd.to_datetime(df['DATETIME'])

# daily_df = df.resample('D', on='DATETIME').last().dropna().reset_index()

# daily_df.to_csv('Data/GBPEUR_daily.csv', index=False)

# print("Hourly data converted to daily close.")
Gold_RC_data = pd.read_csv("Data/GOLD_daily.csv")
Gold_RC_data['DATETIME'] = pd.to_datetime(Gold_RC_data['DATETIME'])

# Step 2: Filter rows where the date is from 1980 onwards
Gold_RC_data_filtered = Gold_RC_data[Gold_RC_data['DATETIME'] >= '1980-01-01']

# Step 3: Overwrite the original file by saving the filtered data with the same filename
Gold_RC_data_filtered.to_csv('Data/GOLD_daily.csv', index=False)

print("Data in 'GOLD_daily.csv' updated successfully.")