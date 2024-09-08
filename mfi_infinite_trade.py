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
import time
import concurrent.futures
import signal
import sys
from multiprocessing import current_process, Manager
from mfi_functions import setup_logging, calculate_mfi, \
                            find_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity, termination_flag, get_exchange_client
from mfi_analysis import mfi_analysis_main

def run_mfi_trading_algo_wrapper(**kwargs):
    # wrapper to apply different logging in this subprocess/thread
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"
    setup_logging(log_dir = out_directory_name, file_suffix=f"{kwargs["symbol"]}_", log_to_stdout=False)
    return run_mfi_trading_algo(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--usdt_amount", required=True, help="USDT amount to operate with. Will be translated into corresponding asset's quantity", type=float)
    parser.add_argument("--pnl_threshold", default=2, type=float)
    parser.add_argument("--liq_threshold", default=0, type=float)
    parser.add_argument("--n_assets_to_trade", default=10, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no_vol_threshold", action="store_true")
    parser.add_argument("--symbols", default=None, type=str)
    parser.add_argument("--exchange", required=False, default="binance")

    args = parser.parse_args()

    symbols = None
    if args.symbols is not None:
        symbols = args.symbols.split(",")

    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    setup_logging(log_dir=out_directory_name, file_suffix=f"infinite_trade")
    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    exchange_client = get_exchange_client(args.exchange)

    logging.info(f"Started infinite trade script")
    iteration = 0
    profits = []
    profits_minus_fees = []
    total_profit = 0
    total_profit_minus_fees = 0

    while True:
        if termination_flag.value:
            logging.info(f"Termination flag is set, terminating")
            break

        logging.info(f"Iteration {iteration}")

        # since it's "infinite", it will run for many days
        # will be more convenient to store result by day
        out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"
        if not os.path.exists(out_directory_name):
            os.makedirs(out_directory_name)

        logging.info(f"Starting analysis")
        logging.disable(logging.WARNING) # to avoid logging a lot of infos
        analysis_results, analysis_df = mfi_analysis_main(exchange_client=exchange_client,
                                                          symbols=symbols,
                                                          no_vol_threshold=args.no_vol_threshold)
        logging.disable(logging.NOTSET)

        analysis_df_sub = analysis_df[(analysis_df.pnl >= args.pnl_threshold) & (analysis_df.liquidity_score > args.liq_threshold)]
        analysis_df_sub = analysis_df_sub.sort_values(by='total_profit', ascending=False)
        chosen_assets = list(analysis_df_sub["symbol"])[:min(analysis_df_sub.shape[0], args.n_assets_to_trade)]

        if len(chosen_assets) == 0:
            logging.info(f"Analysis finished, no assets chosen")
            time.sleep(60 * 60) # sleep for an hour if there are no good assets to trade on
            continue

        logging.info(f"Analysis finished, chosen assets: {chosen_assets}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_assets_to_trade) as executor:
            futures = []
            for asset in chosen_assets:
                futures.append(executor.submit(run_mfi_trading_algo_wrapper, 
                                            symbol=asset,
                                            usdt_amount=args.usdt_amount,
                                            exchange_client=exchange_client,
                                            dry_run=args.dry_run,
                                            out_dir=out_directory_name))
            
            logging.info(f"Waiting for results")
            
            # Wait for all futures to complete and gather results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        total_profit += np.sum([result["total_profit"] for result in results])
        total_profit_minus_fees += np.sum([result["total_profit_minus_fees"] for result in results])
        logging.info(f"Total profit: {total_profit}, minus fees: {total_profit_minus_fees} USDT")
        profits.append(total_profit)
        profits_minus_fees.append(total_profit_minus_fees)
        
        iteration += 1
