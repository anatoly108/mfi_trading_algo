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
import time
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

def process_symbol(args, symbol, exchange_client, out_directory_name, start_date, end_date):
    setup_logging(log_dir = out_directory_name, file_suffix=f"{symbol}_", log_to_stdout=True)
    logging.info(f"Starting {symbol}")
    start_time = time.time()

    try:
        candles = get_candles(symbol=symbol,
                            interval="1m",
                            exchange_client=exchange_client,
                            startTime=start_date,
                            endTime=end_date)
        if len(candles) == 0:
            logging.warning(f"{symbol} no candles")
            return

        df = pd.DataFrame(candles)
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df.to_csv(f"{out_directory_name}/{symbol}.csv", index=False)
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f"Finished {symbol}, time elapsed: {total_time:.2f} minutes")

    except Exception as e:
        # if unexpected errors happen: log them, but don't break the program; we have enough data from other symbols
        logging.error(f"Symbol {symbol}, an error occurred: {e.__class__.__name__}: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--symbols", default=None, type=str)
    parser.add_argument("--months_back", default=6, type=int)
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--exchange", required=False, default="binance")
    
    args = parser.parse_args()

    exchange_client = get_exchange_client(args.exchange)

    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/get_candles/{filename_suffix}_{exchange_client.__class__.__name__}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    setup_logging(log_dir = out_directory_name)

    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    symbols = None
    if args.symbols is not None:
        # take symbols given by user in the command line
        symbols = args.symbols.split(",")

    if symbols is None:
        # take all symbols from exchange
        symbols = exchange_client.get_all_spot_usdt_pairs()

    # write to file which symbols it's operating on
    with open(f"{out_directory_name}/symbols.txt", 'w') as file:
        for symbol in symbols:
            file.write(f"{symbol}\n")
    
    end_date = get_last_complete_time_for_candles("1m")
    start_date = end_date - timedelta(hours=args.months_back * 30 * 24) # simplistic: assume month has 30 days 

    with Manager() as manager:
        semaphore = manager.BoundedSemaphore(1) # allow only that many processes at a time to make a request
        exchange_client.semaphore = semaphore

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.threads) as executor:
            futures = []
            for i, symbol in enumerate(symbols):
                futures.append(executor.submit(process_symbol, 
                                            args=args, 
                                            symbol=symbol, 
                                            exchange_client=exchange_client, 
                                            out_directory_name=out_directory_name,
                                            start_date=start_date,
                                            end_date=end_date))
            
            results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), 
                                                          total=len(futures), 
                                                          position=0,
                                                          desc="Total")]
