import pandas as pd
import numpy as np
import talib as ta
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.signal import find_peaks
import argparse
import os
from mfi_functions import setup_logging, calculate_mfi, \
                            find_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity, ExchangeClient

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

    usdt = 1000
    quantity = usd_to_quantity(usdt, candles[-1][4]) # latest close price to figure out quantity, assume $1k trades
    trading_results = run_mfi_trading_algo(symbol = symbol, 
                                           quantity = quantity, 
                                           dry_run = True,
                                           candles = candles_part1,
                                           get_new_candles_function = get_new_candles_for_analysis,
                                           exit_after_no_candle = True,
                                           do_plot = False)
    buy_signals = trading_results["buy_signals"]
    sell_signals = trading_results["sell_signals"]
    
    ticker_data = ExchangeClient.get_ticker_data(symbol)

    trades_num = len(buy_signals) * 2
    fee_per_trade = 0.075 / 100 # 0.075% fee per trade on level 1
    fees = fee_per_trade*trades_num*usdt
    total_profit_minus_fees = trading_results["total_profit"] - fees

    asset_price_change = round((1 - candles[0][4]/candles[-1][4]) * 100, 1)

    result_dict = {
        "symbol": symbol,
        "candles": candles,
        "mfi": mfi,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "total_profit": round(trading_results["total_profit"], 1),
        "total_profit_minus_fees": round(total_profit_minus_fees, 1),
        "fees": fees,
        "trades_num": trades_num,
        "pnl": round((trading_results["total_profit"]/usdt)*100, 1),
        "asset_price_change": asset_price_change
    }

    # Add all ticker data, but update only keys that are not present in 
    for key, value in ticker_data.items():
        if key not in result_dict:
            result_dict[key] = value

    return result_dict


def mfi_analysis_main(plot_all=False, short=False, symbols=None):
    # Filter symbols that end with 'USDT' and are available for spot trading
    if symbols is None:
        symbols = ExchangeClient.get_all_spot_usdt_pairs()
    # for testing specific symbols
    # symbols = ["AUDIOUSDT"]
    
    print("running analysis")
    results = []
    for symbol in tqdm(symbols):
        result = analyze_pair(symbol)
        if result:
            results.append(result)
    
    subset_results = [
    {
        "symbol": res["symbol"],
        "total_profit": res["total_profit"],
        "fees": res["fees"],
        "total_profit_minus_fees": res["total_profit_minus_fees"],
        "trades_num": res["trades_num"],
        "pnl": res["pnl"],
        "quoteVolume": convert_to_millions(float(res["quoteVolume"])),
        "quoteVolume_raw": float(res["quoteVolume"]),
        "asset_price_change": res["asset_price_change"]
    }
        for res in results
    ]

    # Create a DataFrame from the subset results
    df = pd.DataFrame(subset_results)
    df = df.sort_values(by='total_profit', ascending=False)

    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/analysis/{filename_suffix}"

    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)
    
    df.to_csv(f"{out_directory_name}/{filename_suffix}_crypto_mfi_analysis.csv", index=False)
    df.to_excel(f"{out_directory_name}/{filename_suffix}_crypto_mfi_analysis.xlsx", index=False)

    # Select top 10 assets based on highest total_profit
    top_assets = df[1:min([10, df.shape[0]])]
    flop_assets = df[-min([10, df.shape[0]]):]
    results_top = [res for res in results if res["symbol"] in list(top_assets["symbol"])]
    results_flop = [res for res in results if res["symbol"] in list(flop_assets["symbol"])]

    if plot_all:
        print("making all plots")
        for asset in tqdm(results):
            plot_asset(asset, "_analysis", out_dir=out_directory_name)
    else:
        # Plotting each of the top 10 assets
        for asset in results_top:
            plot_asset(asset, "_analysis_top", out_dir=out_directory_name)

        # Plotting each of the flop 10 assets
        for asset in results_flop:
            plot_asset(asset, "_analysis_flop", out_dir=out_directory_name)

    return results, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--plot_all", action="store_true")
    args = parser.parse_args()

    mfi_analysis_main(args.plot_all)
