import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from binance.client import Client
from scipy.signal import find_peaks
from mfi_functions import load_config, setup_logging, calculate_mfi, \
                            find_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity

def calculate_price_change(candles, minima, maxima):
    changes = []
    for i in range(len(minima)):
        if i < len(maxima):
            min_price = float(candles[minima[i]][4])
            max_price = float(candles[maxima[i]][4])
            changes.append((max_price - min_price) / min_price * 100)
    return changes

def convert_to_millions(volume):
    # Convert the volume to millions
    volume_in_millions = volume / 1_000_000
    
    # Format the number with one decimal place if it's not an integer
    if volume_in_millions >= 1:
        # Use no decimals if it's a whole number
        return f"{volume_in_millions:.0f}M" if volume_in_millions.is_integer() else f"{volume_in_millions:.1f}M"
    else:
        # For numbers less than 1M, show one decimal
        return f"{volume_in_millions:.1f}M"


def analyze_pair(symbol):
    candles = get_candles(symbol, "1m")
    if len(candles) == 0:
        return None
    
    # first, let's calculate perfect extrema to know how a perfect trading would look like
    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)
    troughs, peaks = find_extrema(mfi)

    # next, emulate a situation where second half of candles is unknown
    part1_of_candles_num = round(len(candles)/2)
    part2_of_candles_num = part1_of_candles_num + 1
    candles_part1 = candles[:part1_of_candles_num]
    candles_part2 = candles[part2_of_candles_num:]
    # just get the next candle from candles_part2
    get_new_candles_for_analysis = lambda symbol, interval, last_candle_timestamp: next(
        ([candle] for candle in candles_part2 if candle[0] > last_candle_timestamp), 
        []
    )

    quantity = usd_to_quantity(1000, float(candles[-1][4])) # latest close price to figure out quantity, assume $1k trades
    trading_results = run_mfi_trading_algo(symbol = symbol, 
                                           quantity = quantity, 
                                           config_path = None, 
                                           dry_run = True,
                                           candles = candles_part1,
                                           get_new_candles_function = get_new_candles_for_analysis,
                                           exit_after_no_candle = True,
                                           do_plot = False)
    buy_signals = trading_results["buy_signals"]
    sell_signals = trading_results["sell_signals"]
    
    optimal_changes = calculate_price_change(candles, troughs, peaks)
    real_time_changes = calculate_price_change(candles, buy_signals, sell_signals)

    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)
    ticker_data = response.json()

    result_dict = {
        "symbol": symbol,
        "optimal_sum_change": sum(optimal_changes),
        "real_time_sum_change": sum(real_time_changes),
        "candles": candles,
        "mfi": mfi,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "total_profit": trading_results["total_profit"]
    }

    # Add all ticker data, but update only keys that are not present in 
    for key, value in ticker_data.items():
        if key not in result_dict:
            result_dict[key] = value

    return result_dict


def main():
    client = Client()
    # Fetch exchange information
    exchange_info = client.get_exchange_info()

    # Filter symbols that end with 'USDT' and are available for spot trading
    symbols = [
        symbol['symbol'] 
        for symbol in exchange_info['symbols'] 
        if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']
    ]
    # for testing specific symbols
    # symbols = ["AUDIOUSDT"]
    
    results = []
    for symbol in tqdm(symbols):
        result = analyze_pair(symbol)
        if result:
            results.append(result)
    
    subset_results = [
    {
        "symbol": res["symbol"],
        "optimal_sum_change": res["optimal_sum_change"],
        "real_time_sum_change": res["real_time_sum_change"],
        "total_profit": res["total_profit"],
        "quoteVolume": convert_to_millions(float(res["quoteVolume"]))
    }
        for res in results
    ]

    # Create a DataFrame from the subset results
    df = pd.DataFrame(subset_results)
    df = df.sort_values(by='real_time_sum_change', ascending=False)

    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}"

    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df.to_csv(f"{out_directory_name}/{filename_suffix}_crypto_mfi_analysis.csv", index=False)
    df.to_excel(f"{out_directory_name}/{filename_suffix}_crypto_mfi_analysis.xlsx", index=False)

    # Select top 10 assets based on highest real_time_sum_change
    top_assets = df[1:min([10, df.shape[0]])]
    flop_assets = df[-min([10, df.shape[0]]):]
    results_top = [res for res in results if res["symbol"] in list(top_assets["symbol"])]
    results_flop = [res for res in results if res["symbol"] in list(flop_assets["symbol"])]

    # Plotting each of the top 10 assets
    for asset in results_top:
        plot_asset(asset, "_analysis_top", out_dir=out_directory_name)

    # Plotting each of the flop 10 assets
    for asset in results_flop:
        plot_asset(asset, "_analysis_flop", out_dir=out_directory_name)

if __name__ == "__main__":
    main()
