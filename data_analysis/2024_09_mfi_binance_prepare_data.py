import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Specify the directory containing the CSV files
directory = '/Users/anatoly/projects/trading/out/binance_grand_analysis_6months_4hours_merged'

# List to store each dataframe
dataframes = []

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv') and 'results' not in filename and 'trades' not in filename:
        filepath = os.path.join(directory, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)
        
        # Sort the DataFrame by the 'timepoint' column in ascending order
        df = df.sort_values(by='timepoint', ascending=True)
        
        # Create a new column 'total_profit_future_timeframe' with the next row's 'total_profit'
        df['total_profit_future_timeframe'] = df['total_profit'].shift(-1)
        
        # Append the processed DataFrame to the list
        dataframes.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df = combined_df[combined_df["total_profit_future_timeframe"].notna()]

processed_symbols = np.unique(combined_df['symbol'])
len(processed_symbols)
spearmanr(combined_df["asset_price_change"],
          combined_df["total_profit_future_timeframe"])

# Print or return the combined dataframe
print(combined_df)


# ----
# get_candles checks
df = pd.read_csv("/Users/anatoly/projects/trading/out/2024_09_15/get_candles/2024_09_15_13_55_38_Binance/ETHUSDT.csv")
df['next_time'] = df['time'].shift(-1)
df["time_diff"] = df["next_time"] - df["time"]
df.query("time_diff != 60000")
df.iloc[257804]