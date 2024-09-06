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
from mfi_functions import setup_logging, calculate_mfi, \
                            find_extrema, plot_asset, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity
from mfi_analysis import main

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to the YAML config file containing API keys")
    parser.add_argument("--usdt_amount", required=True, help="USDT amount to operate with. Will be translated into corresponding asset's quantity", type=float)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--pnl_threshold", default=5, type=float)
    parser.add_argument("--n_assets_to_trade", default=3, type=int)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/"
    filename_suffix = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    setup_logging(f"infinite_trade_{filename_suffix}")

    logging.info(f"Started infinite trade script")
    iteration = 0
    profits = []

    while True:
        logging.info(f"Iteration {iteration}")

        logging.info(f"Starting analysis")
        logging.disable(logging.WARNING) # to avoid logging a lot of infos
        analysis_results, analysis_df = mfi_analysis_main(symbols=args.symbols)
        logging.disable(logging.INFO)

        analysis_df_sub = analysis_df[analysis_df.pnl > args.pnl_threshold]
        analysis_df_sub = analysis_df_sub.sort_values(by='total_profit', ascending=False)
        chosen_assets = list(analysis_df_sub["symbol"])[:min(analysis_df_sub.shape[0], args.n_assets_to_trade)]

        if len(chosen_assets) == 0:
            logging.info(f"Analysis finished, no assets chosen")
            time.sleep(60 * 60) # sleep for an hour if there are no good assets to trade on
            continue

        logging.info(f"Analysis finished, chosen assets: {chosen_assets}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the function with different arguments
            futures = []
            for asset in chosen_assets:
                # TODO: specify the necessary arguments
                futures.append(executor.submit(run_mfi_trading_algo, 
                                               usdt_amount=args.usdt_amount,
                                               dry_run=args.dry_run))
            
            # Wait for all futures to complete and gather results
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # TODO: collect profits from all results, report them
        results