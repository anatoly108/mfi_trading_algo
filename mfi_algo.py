import time
import logging
import argparse
import yaml
import os
from datetime import datetime, timedelta
import talib as ta
import numpy as np
import requests
import sys
from mfi_functions import setup_logging, run_mfi_trading_algo, usd_to_quantity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to the YAML config file containing API keys")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--quantity", required=False, help="Quantity to operate with", default=None, type=float)
    parser.add_argument("--usdt_amount", required=False, help="USDT amount to operate with. Will be translated into corresponding asset's quantity", default=None, type=float)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    setup_logging(f"{args.symbol}_")

    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    if args.quantity is None and args.usdt_amount is None:
        logging.error("Either quantity or usdt_amount have to be specified")
        exit(1)

    out_directory_name = f"out/{datetime.now().strftime('%Y_%m_%d')}/trading/"

    # Create the directory if it doesn't exist
    if not os.path.exists(out_directory_name):
        os.makedirs(out_directory_name)

    run_mfi_trading_algo(symbol=args.symbol, 
                         quantity=args.quantity, 
                         usdt_amount=args.quantity, 
                         dry_run=args.dry_run, 
                         out_dir=out_directory_name)

