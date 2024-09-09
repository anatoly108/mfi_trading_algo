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
                            run_mfi_trading_algo, usd_to_quantity, VOL_THRESHOLD, \
                            calculate_liquidity_score, get_exchange_client, write_trading_results, \
                            MFI_TRADING_TIMEOUT_H, LOOKBACK_PERIOD_H

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

def calculate_range_bound_score(candles):
    """
    Calculate how range-bound an asset is based on its 1m candles from the past 24h.
    The score will be high if the price tends to revert to its mean after deviations.
    
    :param candles: List of candles, where each candle is in the format [time, open, high, low, close, volume].
    :return: A score between 0 (not range-bound) and 1 (perfectly range-bound).
    """
    # Extract close prices from the candles
    close_prices = np.array([candle[4] for candle in candles])
    
    # Calculate the mean close price
    mean_close = np.mean(close_prices)
    
    # Calculate deviations from the mean
    deviations = close_prices - mean_close
    
    # Now, check if the price moves back toward the mean after moving away
    reversion_count = 0
    for i in range(1, len(deviations)):
        # If the price was further from the mean and now it's closer, count as a reversion
        if abs(deviations[i]) < abs(deviations[i-1]):
            reversion_count += 1
    
    # Normalize the reversion count by the total number of candles (1440 in 24h data)
    reversion_ratio = reversion_count / (len(candles) - 1)
    
    # The range-bound score is the reversion ratio (higher means more range-bound)
    return reversion_ratio

def calculate_volatility_range(candles):
    """
    Calculate the volatility range of an asset based on its 1m candles.
    Higher score means more volatile behavior.
    
    :param candles: List of candles, where each candle is in the format [time, open, high, low, close, volume].
    :return: A score between 0 (low volatility) and 1 (high volatility).
    """
    # Extract close prices from the candles
    close_prices = np.array([candle[4] for candle in candles])
    
    # Calculate percentage price changes between consecutive candles
    percent_changes = np.abs(np.diff(close_prices) / close_prices[:-1]) * 100
    
    # Sum the absolute percentage changes to get the total volatility
    total_volatility = np.sum(percent_changes)
    
    # Determine the maximum expected volatility based on the number of candles
    number_of_candles = len(candles)
    max_volatility_per_candle = 1  # Example: assume 1% per candle is extreme
    max_expected_volatility = number_of_candles * max_volatility_per_candle
    
    # Normalize the total volatility to a 0-1 scale
    volatility_score = min(total_volatility / max_expected_volatility, 1)
    
    return volatility_score


def analyze_pair(ticker_data, exchange_client, now=None, do_calculate_liquidity_score=True):
    symbol = ticker_data["symbol"]
    candles = get_candles(symbol=symbol, 
                          interval="1m", 
                          exchange_client=exchange_client, 
                          hours=LOOKBACK_PERIOD_H+MFI_TRADING_TIMEOUT_H,
                          now=now)
    if len(candles) == 0:
        return None
    
    # first, let's calculate perfect extrema to know how a perfect trading would look like
    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)
    troughs, peaks = find_extrema(mfi)

    # next, emulate a situation where only LOOKBACK_PERIOD_H candles is known
    part1_of_candles_num = LOOKBACK_PERIOD_H * 60
    part2_of_candles_num = part1_of_candles_num + 1 # MFI_TRADING_TIMEOUT_H * 60
    candles_part1 = candles[:part1_of_candles_num]
    candles_part2 = candles[part2_of_candles_num:]
    # just get the next candle from candles_part2
    get_new_candles_for_analysis = lambda symbol, interval, last_candle_timestamp, exchange_client: next(
        ([candle] for candle in candles_part2 if candle[0] > last_candle_timestamp), 
        []
    )

    usdt = 1000
    quantity = usd_to_quantity(usdt, candles[-1][4]) # latest close price to figure out quantity, assume $1k trades
    trading_results = run_mfi_trading_algo(symbol = symbol, 
                                           quantity = quantity, 
                                           exchange_client = exchange_client,
                                           dry_run = True,
                                           candles = candles_part1,
                                           get_new_candles_function = get_new_candles_for_analysis,
                                           exit_after_no_candle = True,
                                           do_plot = False)
    buy_signals = trading_results["buy_signals"]
    sell_signals = trading_results["sell_signals"]

    trades_num = len(buy_signals) * 2
    fee_per_trade = exchange_client.get_taker_fee_fraction()
    fees = fee_per_trade*trades_num*usdt
    total_profit_minus_fees = trading_results["total_profit"] - fees

    asset_price_change = round((1 - candles[0][4]/candles[-1][4]) * 100, 1)
    
    range_bound_score = calculate_range_bound_score(candles_part1)
    volatility_score = calculate_volatility_range(candles_part1)

    result_dict = {
        "symbol": symbol,
        "candles": candles,
        "mfi": mfi,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "profits": trading_results["profits"],
        "total_profit": round(trading_results["total_profit"], 1),
        "total_profit_minus_fees": round(total_profit_minus_fees, 1),
        "fees": fees,
        "trades_num": trades_num,
        "pnl": round((trading_results["total_profit"]/usdt)*100, 1),
        "asset_price_change": asset_price_change,
        "range_bound_score": range_bound_score,
        "volatility_score": volatility_score
    }

    if do_calculate_liquidity_score:
        liquidity_score = calculate_liquidity_score(symbol=symbol, exchange_client=exchange_client)
        result_dict["liquidity_score"] = liquidity_score

    # Add all ticker data, but update only keys that are not present in 
    for key, value in ticker_data.items():
        if key not in result_dict:
            result_dict[key] = value

    if "quoteVolume" in result_dict.keys():
        result_dict["quoteVolume"] = float(result_dict["quoteVolume"])

    return result_dict


def mfi_analysis_main(exchange_client, plot_all=False, short=False, symbols=None, no_vol_threshold=False, vol_threshold=VOL_THRESHOLD, now=None):
    if symbols is None:
        symbols = exchange_client.get_all_spot_usdt_pairs()
    
    tickers_final = []

    if no_vol_threshold:
        tickers_final = [{"symbol": symbol} for symbol in symbols]
    else:
        tickers = exchange_client.get_all_ticker_data()
        for symbol in symbols:
            ticker = next((ticker for ticker in tickers if ticker["symbol"] == symbol), None)
            if ticker is None:
                continue

            # 24h volume threshold
            if float(ticker["quoteVolume"]) > vol_threshold or no_vol_threshold: 
                tickers_final.append(ticker)

    print("running analysis")
    results = []
    for ticker in tqdm(tickers_final):
        result = analyze_pair(ticker_data=ticker, 
                              exchange_client=exchange_client, 
                              now=now)
        if result:
            results.append(result)
    
    # results will become a DataFrame, so we only keep simple values, no lists/arrays 
    subset_results = [{key: value for key, value in result.items() if isinstance(value, (str, int, float))} for result in results]

    # Create a DataFrame from the subset results
    df = pd.DataFrame(subset_results)
    df = df.sort_values(by='total_profit', ascending=False)

    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/analysis/{filename_suffix}_{exchange_client.__class__.__name__}"

    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)
    
    df.to_csv(f"{out_directory_name}/{filename_suffix}_{exchange_client.__class__.__name__}_crypto_mfi_analysis.csv", index=False)
    df.to_excel(f"{out_directory_name}/{filename_suffix}_{exchange_client.__class__.__name__}_crypto_mfi_analysis.xlsx", index=False)

    write_trading_results(results=results,
                        global_results_csv=f"{out_directory_name}/{filename_suffix}_{exchange_client.__class__.__name__}_crypto_mfi_analysis_results.csv",
                        global_trades_csv=f"{out_directory_name}/{filename_suffix}_{exchange_client.__class__.__name__}_crypto_mfi_analysis_trades.csv",
                        additional_values_to_add={"exchange": exchange_client.__class__.__name__})
                            
    # Select top 10 assets based on highest total_profit
    top_assets = df[0:min([9, df.shape[0]])]
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
    parser.add_argument("--symbols", default=None, type=str)
    parser.add_argument("--now", default=None, type=str) # in the format: 2024_08_20__18_54 in UTC
    parser.add_argument("--plot_all", action="store_true")
    parser.add_argument("--no_vol_threshold", action="store_true")
    parser.add_argument("--vol_threshold", required=False, default=VOL_THRESHOLD, type=float)
    parser.add_argument("--exchange", required=False, default="binance")
    args = parser.parse_args()

    exchange_client = get_exchange_client(args.exchange)

    symbols = None
    if args.symbols is not None:
        symbols = args.symbols.split(",")

    now = None
    if args.now is not None:
        now = datetime.strptime(args.now, "%Y_%m_%d__%H_%M").replace(tzinfo=timezone.utc)

    mfi_analysis_main(exchange_client=exchange_client,
                      plot_all=args.plot_all, 
                      no_vol_threshold=args.no_vol_threshold, 
                      vol_threshold=args.vol_threshold,
                      symbols=symbols,
                      now=now)
