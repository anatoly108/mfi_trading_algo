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
import sys
from multiprocessing import BoundedSemaphore, Manager
import concurrent.futures
from mfi_functions import setup_logging, calculate_mfi, \
                            find_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity, VOL_THRESHOLD, \
                            calculate_liquidity_score, get_exchange_client, write_trading_results, \
                            MFI_TRADING_TIMEOUT_H, LOOKBACK_PERIOD_H, convert_to_unix, get_last_complete_time_for_candles
from mfi_analysis import analyze_pair

def generate_timepoints(start_date, end_date, hours):
    """
    Generates intervals of a specified number of hours from end_date back to start_date.
    
    :param start_date: The earliest allowed date.
    :param end_date: The latest date to start from.
    :param hours: The number of hours for each interval.
    :return: List of datetime objects representing the intervals.
    """
    timepoints = []
    current_time = end_date
    
    # Keep generating intervals while the current time is greater than start_date
    # important: timepoints are generated from most recent to oldest
    while current_time > start_date:
        timepoints.append(current_time)
        # Subtract the specified number of hours from the current time
        current_time -= timedelta(hours=hours)
    
    return timepoints

def process_symbol(args, symbol, exchange_client, out_directory_name, bar_pos):
    setup_logging(log_dir = out_directory_name, file_suffix=f"{symbol}_", log_to_stdout=False)

    try:
        end_date = get_last_complete_time_for_candles("1m")
        start_date = end_date - timedelta(hours=args.months_back * 30 * 24) # simplistic: assume month has 30 days 

        # important: timepoints are generated from most recent to oldest
        timepoints = generate_timepoints(start_date, end_date, MFI_TRADING_TIMEOUT_H)
        all_timepoint_results = []

        for timepoint in tqdm(timepoints, desc=f"Timepoints for {symbol}", position=bar_pos, leave=True):
            timepoint_results = analyze_pair(ticker_data={"symbol": symbol},
                                            exchange_client=exchange_client,
                                            now=timepoint,
                                            do_calculate_liquidity_score=False)
            if timepoint_results is None:
                # not enough candles to cover history that far back
                # that's where it's important that timepoints are generated from most recent to oldest
                break

            # timepoint_results is the "input" data that we use to trade next MFI_TRADING_TIMEOUT_H hours
            # now, we need "output" data which is the trading results of the next MFI_TRADING_TIMEOUT_H hours
            # we'll sumply get it with next timepoint_results because it will be MFI_TRADING_TIMEOUT_H shifted
            # therefore it becomes a loop: every result is "output" of previous and "input" for next
            timepoint_results["timepoint"] = convert_to_unix(timepoint)

            # all_timepoint_results will become a DataFrame, so we only keep simple values, no lists/arrays 
            timepoint_results_sub = {key: value for key, value in timepoint_results.items() if isinstance(value, (str, int, float))}
            all_timepoint_results.append(timepoint_results_sub)

        df = pd.DataFrame(all_timepoint_results)
        df.to_csv(f"{out_directory_name}/{symbol}.csv", index=False)
    except Exception as e:
        # if unexpected errors happen: log them, but don't break the program; we have enough data from other symbols
        logging.error(f"Symbol {symbol}, an error occurred: {e.__class__.__name__}: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--symbols", default=None, type=str)
    parser.add_argument("--months_back", default=6, type=int)
    parser.add_argument("--threads", default=os.cpu_count(), type=int)
    parser.add_argument("--exchange", required=False, default="binance")
    
    args = parser.parse_args()

    exchange_client = get_exchange_client(args.exchange)

    symbols = None
    if args.symbols is not None:
        symbols = args.symbols.split(",")

    if symbols is None:
        symbols = exchange_client.get_all_spot_usdt_pairs()

    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/grand_analysis/{filename_suffix}_{exchange_client.__class__.__name__}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    setup_logging(log_dir = out_directory_name)

    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))
    
    logging.disable(logging.INFO) # to avoid logging a lot of infos

    with Manager() as manager:
        semaphore = manager.BoundedSemaphore(2) # allow only that many processes at a time to make a request
        exchange_client.semaphore = semaphore

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = []
            for i, symbol in enumerate(symbols):
                futures.append(executor.submit(process_symbol, 
                                            args=args, 
                                            symbol=symbol, 
                                            exchange_client=exchange_client, 
                                            out_directory_name=out_directory_name,
                                            bar_pos=i+1))
            
            results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), 
                                                          total=len(futures), 
                                                          position=0,
                                                          desc="Total")]
