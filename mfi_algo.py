import time
import logging
import argparse
import yaml
import os
from binance.client import Client
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
    parser.add_argument("--quantity", required=False, help="Quantity to operate with")
    parser.add_argument("--usdt_amount", required=False, help="USDT amount to operate with. Will be translated into corresponding asset's quantity")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    setup_logging(f"{args.symbol}_")

    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    if args.quantity is None and args.usdt_amount is None:
        logging.error("Either quantity or usdt_amount have to be specified")
        exit(1)

    if args.usdt_amount is not None:
        client = Client()
        ticker = client.get_symbol_ticker(symbol=args.symbol)
        current_price = float(ticker['price'])
        client.close_connection()
        quantity = usd_to_quantity(float(args.usdt_amount), current_price)
        logging.info(f"Chosen quantity is: {quantity}, equivalent to {current_price * quantity} USDT")

    if args.quantity is not None:
        quantity = float(args.quantity)

    run_mfi_trading_algo(args.symbol, quantity, args.config, args.dry_run)
