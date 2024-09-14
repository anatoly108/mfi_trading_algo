import pandas as pd
import numpy as np
import talib
import requests
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.signal import find_peaks
import argparse
import os
import sys
from scipy import stats
from multiprocessing import BoundedSemaphore, Manager
import concurrent.futures
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

def calculate_emas(candles):
    """
    Calculate EMA200, EMA100 from a list of candles.
    
    :param candles: List of candles, where each candle is in the format [time, open, high, low, close, volume].
    """
    # Extract close prices from the candles
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    
    # Calculate EMA200 and EMA100 using TA-Lib
    ema200 = talib.EMA(close_prices, timeperiod=200)
    ema100 = talib.EMA(close_prices, timeperiod=100)

    if ema200 is None or ema100 is None:
        return None, None, None, None, \
            None, None, None, None
    
    # Get the latest EMA values
    ema200_latest = ema200[-1]
    ema100_latest = ema100[-1]
    
    # get fist existing value of each ema
    ema200_start = next((value for value in ema200 if not np.isnan(value)))
    ema100_start = next((value for value in ema200 if not np.isnan(value)))
    
    # Normalize EMA values
    min_price = np.min(close_prices)
    max_price = np.max(close_prices)

    # Avoid division by zero
    price_range = max_price - min_price
    if price_range == 0:
        ema100_latest_normalized = 0
        ema200_latest_normalized = 0
        ema100_start_normalized = 0
        ema200_start_normalized = 0
    else:
        # Normalize EMA values to range 0-1
        ema100_latest_normalized = (ema100_latest - min_price) / price_range
        ema200_latest_normalized = (ema200_latest - min_price) / price_range
        ema100_start_normalized = (ema100_start - min_price) / price_range
        ema200_start_normalized = (ema200_start - min_price) / price_range

    return ema100_start, ema100_latest, ema100_start_normalized, ema100_latest_normalized, \
            ema200_start, ema200_latest, ema200_start_normalized, ema200_latest_normalized

def calculate_win_rate(profits):
    winning_trades = [profit for profit in profits if profit > 0]
    total_trades = len(profits)
    if total_trades == 0:
        return 0.0
    win_rate = len(winning_trades) / total_trades
    return win_rate * 100  # Return as a percentage

def calculate_average_trade_profit(total_profit, total_trades):
    if total_trades == 0:
        return 0.0
    average_profit = total_profit / total_trades
    return average_profit

def calculate_max_drawdown(profits):
    if len(profits) < 2:
        return 0.0
    cumulative_profits = np.cumsum(profits)
    running_max = np.maximum.accumulate(cumulative_profits)
    drawdowns = running_max - cumulative_profits
    max_drawdown = drawdowns.max()
    return max_drawdown

def calculate_sharpe_ratio(profits):
    if len(profits) < 2:
        return 0.0
    average_return = np.mean(profits)
    return_std = np.std(profits)
    if return_std == 0:
        return 0.0
    sharpe_ratio = average_return / return_std
    return sharpe_ratio

def calculate_profit_factor(profits):
    gross_profit = sum([profit for profit in profits if profit > 0])
    gross_loss = -sum([profit for profit in profits if profit < 0])
    if gross_loss == 0:
        return float('inf')  # Infinite profit factor
    profit_factor = gross_profit / gross_loss
    return profit_factor

def calculate_average_holding_time_and_time_in_market(buy_signals, sell_signals, candles):
    holding_times = []
    total_holding_time = 0
    for buy, sell in zip(buy_signals, sell_signals):
        holding_time = sell - buy
        holding_times.append(holding_time)
        total_holding_time += holding_time
    if len(holding_times) == 0:
        return 0.0, 0.0
    average_holding_time = np.mean(holding_times)
    time_in_market = total_holding_time / len(candles)
    return average_holding_time, time_in_market

def calculate_atr(candles, period=14):
    high_prices = np.array([candle[2] for candle in candles], dtype=float)
    low_prices = np.array([candle[3] for candle in candles], dtype=float)
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
    return atr[-1] if atr.size > 0 else 0.0

def calculate_rsi(candles, period=14):
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    rsi = talib.RSI(close_prices, timeperiod=period)
    return rsi[-1] if rsi.size > 0 else 0.0

def calculate_bollinger_bands_width(candles, period=20, num_std_dev=2):
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    upper_band, middle_band, lower_band = talib.BBANDS(
        close_prices, timeperiod=period, nbdevup=num_std_dev, nbdevdn=num_std_dev, matype=0)
    if middle_band[-1] == 0:
        return 0.0
    bb_width = (upper_band[-1] - lower_band[-1]) / middle_band[-1]
    return bb_width

def calculate_rate_of_change(candles, period=14):
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    roc = talib.ROCP(close_prices, timeperiod=period)
    return roc[-1] * 100 if roc.size > 0 else 0.0  # Convert to percentage

def calculate_std_dev_returns(profits):
    if len(profits) < 2:
        return 0.0
    std_dev = np.std(profits)
    return std_dev

def calculate_skewness(profits):
    if len(profits) < 3:
        return 0.0
    skewness = stats.skew(profits)
    return skewness

def calculate_kurtosis(profits):
    if len(profits) < 4:
        return 0.0
    kurtosis = stats.kurtosis(profits)
    return kurtosis

def calculate_vwap(candles):
    high_prices = np.array([candle[2] for candle in candles], dtype=float)
    low_prices = np.array([candle[3] for candle in candles], dtype=float)
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    volumes = np.array([candle[5] for candle in candles], dtype=float)
    typical_prices = (high_prices + low_prices + close_prices) / 3
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return 0.0
    vwap = np.sum(typical_prices * volumes) / total_volume
    return vwap

def calculate_average_daily_volume(candles, period_days=1):
    volumes = np.array([candle[5] for candle in candles], dtype=float)
    period_candles = period_days * 1440  # Assuming 1-minute candles
    if len(volumes) < period_candles:
        period_candles = len(volumes)
    avg_volume = np.mean(volumes[-period_candles:])
    return avg_volume

def calculate_volume_volatility(candles, period_days=1):
    volumes = np.array([candle[5] for candle in candles], dtype=float)
    period_candles = period_days * 1440
    if len(volumes) < period_candles:
        period_candles = len(volumes)
    volume_std_dev = np.std(volumes[-period_candles:])
    return volume_std_dev

def calculate_macd(candles, fast_period=12, slow_period=26, signal_period=9):
    close_prices = np.array([candle[4] for candle in candles], dtype=float)
    macd, macd_signal, macd_hist = talib.MACD(
        close_prices, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    return macd[-1], macd_signal[-1], macd_hist[-1]

def analyze_pair(ticker_data, exchange_client, now=None, do_calculate_liquidity_score=True):
    symbol = ticker_data["symbol"]
    candles = get_candles(symbol=symbol, 
                          interval="1m", 
                          exchange_client=exchange_client, 
                          hours=LOOKBACK_PERIOD_H+MFI_TRADING_TIMEOUT_H,
                          now=now)
    if len(candles) == 0:
        # no candles for the period
        return None
    
    if len(candles) < (LOOKBACK_PERIOD_H+MFI_TRADING_TIMEOUT_H) * 60:
        # not enough candles: history doesn't go that far back
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

    trades_num = len(sell_signals)
    orders_num = len(buy_signals) * 2
    fee_per_trade = exchange_client.get_taker_fee_fraction()
    fees = fee_per_trade*orders_num*usdt
    total_profit_minus_fees = trading_results["total_profit"] - fees

    asset_price_change = round((1 - candles[0][4]/candles[-1][4]) * 100, 1)
    range_bound_score = calculate_range_bound_score(candles)
    volatility_score = calculate_volatility_range(candles)
    # this will be approximate because we can't calculate every single trade here
    # 24 * 60 = 1440 
    candles_past_24h = candles[-1440:] if len(candles) >= 1440 else candles
    quote_volume = np.sum([candle[4] * candle[5] for candle in candles_past_24h])
    ema100_start, ema100_latest, ema100_start_normalized, ema100_latest_normalized, \
    ema200_start, ema200_latest, ema200_start_normalized, ema200_latest_normalized = calculate_emas(candles)

    empty_candles_number = np.sum([candle[5] == 0 for candle in candles_past_24h])
    empty_candles_fraction = empty_candles_number / len(candles_past_24h)
    macd_line, signal_line, macd_histogram = calculate_macd(candles)
    average_holding_time, time_in_market = calculate_average_holding_time_and_time_in_market(buy_signals, sell_signals, candles=candles_part2)

    result_dict = {
        "symbol": symbol,
        "candles": candles,
        "mfi": mfi,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "profits": trading_results["profits"],
        "total_profit": round(trading_results["total_profit"], 1),
        "total_profit_minus_fees": round(total_profit_minus_fees, 1),
        "pnl": round((trading_results["total_profit"]/usdt)*100, 1),
        "fees": fees,
        "orders_num": orders_num,
        "trades_num": trades_num,
        "asset_price_change": asset_price_change,
        "range_bound_score": range_bound_score,
        "volatility_score": volatility_score,
        "quote_volume": quote_volume,
        "empty_candles_number": empty_candles_number,
        "win_rate": calculate_win_rate(trading_results["profits"]),
        "average_trade_profit": calculate_average_trade_profit(trading_results["total_profit"], trades_num),
        "max_drawdown": calculate_max_drawdown(trading_results["profits"]),
        "sharpe_ratio": calculate_sharpe_ratio(trading_results["profits"]),
        "profit_factor": calculate_profit_factor(trading_results["profits"]),
        "average_holding_time": average_holding_time,
        "time_in_market": time_in_market,
        "atr": calculate_atr(candles),
        "rsi": calculate_rsi(candles),
        "bb_width": calculate_bollinger_bands_width(candles),
        "roc": calculate_rate_of_change(candles),
        "std_dev_returns": calculate_std_dev_returns(trading_results["profits"]),
        "skewness": calculate_skewness(trading_results["profits"]),
        "kurtosis": calculate_kurtosis(trading_results["profits"]),
        "vwap": calculate_vwap(candles),
        "average_daily_volume": calculate_average_daily_volume(candles),
        "volume_volatility": calculate_volume_volatility(candles),
        "macd_line": macd_line, 
        "signal_line": signal_line, 
        "macd_histogram": macd_histogram,
        "empty_candles_fraction": empty_candles_fraction,
        "ema100_start": ema100_start,
        "ema100_latest": ema100_latest,
        "ema100_start_normalized": ema100_start_normalized,
        "ema100_latest_normalized": ema100_latest_normalized,
        "ema200_start": ema200_start,
        "ema200_latest": ema200_latest,
        "ema200_start_normalized": ema200_start_normalized,
        "ema200_latest_normalized": ema200_latest_normalized
    }

    if do_calculate_liquidity_score:
        try:
            liquidity_score = calculate_liquidity_score(symbol=symbol, exchange_client=exchange_client)
            result_dict["liquidity_score"] = liquidity_score
        except Exception as e:
            # liquidity score is the only function that deals with order book,
            # and apparently unexpected errors might occur in connection with order books
            # liquidity_score doesn't play important role as of now, so we only calculate it if it's possible
            logging.error(f"Symbol {symbol}, calculate_liquidity_score error: {e.__class__.__name__}: {e}")
            result_dict["liquidity_score"] = None

    # Add all ticker data, but update only keys that are not present in 
    for key, value in ticker_data.items():
        if key not in result_dict:
            result_dict[key] = value

    # this will be quoteVolume reported by exchange
    if "quoteVolume" in result_dict.keys():
        result_dict["quoteVolume"] = float(result_dict["quoteVolume"])

    return result_dict

def get_suffix_and_out_directory_name(exchange_client):
    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/analysis/{filename_suffix}_{exchange_client.__class__.__name__}"
    return filename_suffix, out_directory_name

def mfi_analysis_main(exchange_client, plot_all=False, short=False, symbols=None, no_vol_threshold=False, 
                      vol_threshold=VOL_THRESHOLD, now=None, threads=os.cpu_count(), 
                      filename_suffix=None, out_directory_name=None):
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

    with Manager() as manager:
        semaphore = manager.BoundedSemaphore(4) # allow only that many processes at a time to make a request
        exchange_client.semaphore = semaphore

        logging.info(f"Running analysis in {threads} threads")
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = []
            for ticker in tickers_final:
                futures.append(executor.submit(analyze_pair, 
                                                ticker_data=ticker, 
                                                exchange_client=exchange_client, 
                                                now=now))
            
            results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)) if future.result()]
    
    exchange_client.semaphore = None # remove semaphore: otherwise it will hang indefinitely when manager is closed
    
    # results will become a DataFrame, so we only keep simple values, no lists/arrays 
    subset_results = [{key: value for key, value in result.items() if isinstance(value, (str, int, float))} for result in results]

    # Create a DataFrame from the subset results
    df = pd.DataFrame(subset_results)
    df = df.sort_values(by='total_profit', ascending=False)

    if out_directory_name is None and filename_suffix is None:
        filename_suffix, out_directory_name = get_suffix_and_out_directory_name(exchange_client=exchange_client)

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
        logging.info("Making all plots")
        for asset in tqdm(results):
            plot_asset(asset, "_analysis", out_dir=out_directory_name)
    else:
        logging.info("Making top/flop plots")
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
    parser.add_argument("--threads", default=os.cpu_count(), type=int)
    args = parser.parse_args()

    exchange_client = get_exchange_client(args.exchange)

    filename_suffix, out_directory_name = get_suffix_and_out_directory_name(exchange_client=exchange_client)
    setup_logging(log_dir = out_directory_name)
    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    symbols = None
    if args.symbols is not None:
        symbols = args.symbols.split(",")

    now = None
    if args.now is not None:
        now = datetime.strptime(args.now, "%Y_%m_%d__%H_%M").replace(tzinfo=timezone.utc)

    logging.disable(logging.INFO) # to avoid logging a lot of infos
    mfi_analysis_main(exchange_client=exchange_client,
                      plot_all=args.plot_all, 
                      no_vol_threshold=args.no_vol_threshold, 
                      vol_threshold=args.vol_threshold,
                      symbols=symbols,
                      now=now,
                      threads=args.threads,
                      filename_suffix=filename_suffix,
                      out_directory_name=out_directory_name)
