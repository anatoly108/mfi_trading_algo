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
import json
import xgboost as xgb
import joblib
from multiprocessing import current_process, Manager
from mfi_functions import calculate_mfi, \
                            find_extrema, get_candles, MFI_TIMEINTERVAL, \
                            run_mfi_trading_algo, usd_to_quantity, termination_flag, get_exchange_client, \
                            write_trading_results, VOL_THRESHOLD
from functions import setup_logging
from mfi_analysis import mfi_analysis_main

def run_mfi_trading_algo_wrapper(**kwargs):
    # wrapper to apply different logging in this subprocess/thread
    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"
    setup_logging(log_dir = out_directory_name, file_suffix=f"{kwargs["symbol"]}_", log_to_stdout=False)
    return run_mfi_trading_algo(**kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--usdt_amount", required=True, help="USDT amount to operate with. Will be translated into corresponding asset's quantity", type=float)
    parser.add_argument("--pnl_min", default=0, type=float)
    parser.add_argument("--pnl_max", default=20, type=float)
    parser.add_argument("--liq_min", default=0, type=float)
    parser.add_argument("--liq_max", default=1e6, type=float)
    parser.add_argument("--n_assets_to_trade", default=10, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no_vol_threshold", action="store_true")
    parser.add_argument("--vol_threshold", required=False, default=VOL_THRESHOLD, type=float)
    parser.add_argument("--symbols", default=None, type=str)
    parser.add_argument("--exchange", required=False, default="binance")
    parser.add_argument("--threads", default=os.cpu_count(), type=int)
    parser.add_argument("--empty_candles_fraction", default=0.05, type=float)

    args = parser.parse_args()

    symbols = None
    if args.symbols is not None:
        symbols = args.symbols.split(",")

    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    global_results_csv = f"out/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_infinite_trade_results.csv"
    global_trades_csv = f"out/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_infinite_trade_trades.csv"

    setup_logging(log_dir=out_directory_name, file_suffix=f"infinite_trade")
    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    exchange_client = get_exchange_client(args.exchange)

    logging.info(f"Started infinite trade script")
    iteration = 0
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
                                                          no_vol_threshold=args.no_vol_threshold,
                                                          threads=args.threads,
                                                          vol_threshold=args.vol_threshold)
        logging.disable(logging.NOTSET)

        analysis_df = analysis_df[(analysis_df.volatility_score != 1) &  # volatility_score = 1 are too crazy weird coins
                                 (analysis_df.empty_candles_fraction <= args.empty_candles_fraction)]

        # XGBoost part
        scaler = joblib.load('ml/2024_09_21_binance_6months_2hours_scaler.pkl')
        final_model = joblib.load('ml/2024_09_21_binance_6months_2hours_xgboost.pkl')
        with open('ml/2024_09_21_binance_6months_2hours_values.json', 'r') as f:
            model_values = json.load(f)
        xgboost_columns = model_values["xgboost_columns"]
        xgboost_columns = [i for i in xgboost_columns if i != "Y"]
        
        xgboost_columns.append("symbol")
        X_new = analysis_df[xgboost_columns]
        X_new = X_new.dropna()
        X_new_symbols = list(X_new["symbol"])
        X_new.drop('symbol', axis=1, inplace=True)
        X_new_np = X_new.values
        X_new_scaled = scaler.transform(X_new_np)
        dnew = xgb.DMatrix(X_new_scaled)
        y_pred_proba = final_model.predict(dnew)
        # Class 3 is the positive pnl class; meaning it's column 2
        pos_class_proba = list(y_pred_proba[:,2])
        xgboost_results_df = pd.DataFrame({
            'symbol': X_new_symbols,
            'xgboost_score': pos_class_proba
        })

        analysis_df = pd.merge(analysis_df, xgboost_results_df, on='symbol', how='outer')
        analysis_df = analysis_df[analysis_df["xgboost_score"] > model_values["high_confidence_score"]].reset_index(drop=True)
        analysis_df = analysis_df.sort_values(by='xgboost_score', ascending=False)
        logging.info(f"Number of high confidence assets: {analysis_df.shape[0]}")
        chosen_assets = list(analysis_df["symbol"])[:min(analysis_df.shape[0], args.n_assets_to_trade)]

        if len(chosen_assets) == 0:
            logging.info(f"Analysis finished, no assets chosen")
            time.sleep(60 * 60) # sleep for an hour if there are no good assets to trade on
            continue
        
        logging.info(min(analysis_df["xgboost_score"]))
        logging.info(max(analysis_df["xgboost_score"]))
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

        logging.info(f"Writing results to file")
        write_trading_results(results=results,
                              global_results_csv=global_results_csv,
                              global_trades_csv=global_trades_csv,
                              additional_values_to_add={"iteration": iteration,
                                                        "exchange": exchange_client.__class__.__name__})
        
        iteration += 1
