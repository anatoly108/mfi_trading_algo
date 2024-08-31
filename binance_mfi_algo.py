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
from functions import load_config, setup_logging, get_candles, calculate_mfi, find_extrema, real_time_extrema, plot_asset, MFI_THRESHOLD_LOW, MFI_THRESHOLD_HIGH, MFI_STEP_THRESHOLD, MFI_TIMEINTERVAL

def execute_trade(symbol, quantity, action, config_path, dry_run):
    config = load_config(config_path)
    client = Client(config['api_key'], config['api_secret'])
    
    if dry_run:
        logging.info(f"Dry run {action}")
        return {'price': None}

    if action == 'buy':
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        logging.info(f"Market Buy Order: {order}")
    elif action == 'sell':
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logging.info(f"Market Sell Order: {order}")
    
    client.close_connection()

def main(symbol, quantity, config_path, dry_run, negative_cancel_num=4): 
    start_time = datetime.now()
    
    # Load initial candles and MFI
    candles = get_candles(symbol, "1m", "1440")
    mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)

    last_local_minima = 100
    last_local_maxima = 0
    bought = False
    really_new_candles = []
    total_profit = 0
    buy_signals = []
    sell_signals = []
    profits = []

    for i in range(1, len(mfi)):
        mfi_i = mfi[i]
        
        # minima
        if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
            last_local_minima = mfi_i
    
    while True:
        # Recalculate MFI with the new candle(s)
        mfi = calculate_mfi(candles, MFI_TIMEINTERVAL)
        plot_asset({
            "symbol": symbol,
            "candles": candles,
            "mfi": mfi,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals
        }, "_trading")

        # We have to account for the case when we get >1 new candle
        mfi_new_from = len(mfi) - len(really_new_candles)
        mfi_new_to = len(mfi)
        # if len(really_new_candles) == 0, then this for loop won't even start
        for i in range(mfi_new_from, mfi_new_to):
            mfi_i = mfi[i]
        
            # minima
            if mfi_i < MFI_THRESHOLD_LOW and mfi_i < last_local_minima:
                last_local_minima = mfi_i

            diff_to_minima = mfi_i - last_local_minima 
            if mfi_i < MFI_THRESHOLD_LOW and mfi_i > last_local_minima and diff_to_minima > MFI_STEP_THRESHOLD and not bought:
                last_local_minima = 100
                bought = True
                # buy signal
                order = execute_trade(symbol, quantity, "buy", config_path, dry_run)
                if order["price"] is None:
                    buy_price = float(candles[i][4]) # take last close price for dry run
                else:
                    buy_price = float(order["price"])
                buy_signals.append(i)
                logging.info(f"Buy signal: price {buy_price}, MFI {mfi_i}")
                break # this will break only from the mfi for loop - we can't sell inside this for loop because we would sell for the same price

            if not bought:
                logging.info(f"Not bought, new MFI value: {mfi_i}")
                continue

            # maxima
            if mfi_i > MFI_THRESHOLD_HIGH and mfi_i > last_local_maxima:
                last_local_maxima = mfi_i

            diff_to_maxima = last_local_maxima - mfi_i
            if mfi_i > MFI_THRESHOLD_HIGH and mfi_i < last_local_maxima and diff_to_maxima > MFI_STEP_THRESHOLD:
                last_local_maxima = 0
                bought = False
                # sell signal
                order = execute_trade(symbol, quantity, "sell", config_path, dry_run)

                if order["price"] is None:
                    sell_price = float(candles[i][4]) # take last close price for dry run
                else:
                    sell_price = float(order["price"])

                sell_signals.append(i)
                logging.info(f"Sell signal: price {sell_price}, MFI {mfi_i}")

                sell_final = sell_price * quantity
                buy_final = buy_price * quantity
                
                profit = sell_final - buy_final
                total_profit += profit
                profits.append(profit)
                logging.info(f"Current trade profit: {profit} USDT")
                logging.info(f"Total profit: {total_profit} USDT")

        # Sleep for 60 seconds before fetching new data
        logging.info("Waiting for the next candle")
        time.sleep(60)
        # Get next candle and add it
        # get several candles just to be sure we didn't miss any due to some glitch
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=10"
        response = requests.get(url)
        new_candles = response.json()
        all_current_timestamps = [candle[0] for candle in candles]
        really_new_candles = [candle for candle in new_candles if candle[0] not in all_current_timestamps]
        if len(really_new_candles) == 0:
            # no new candles - can happen
            continue
        logging.info(f"Got {len(really_new_candles)} new candle(s)") 
        candles.extend(really_new_candles)
        
        # Check last N profits
        if len(profits) >= negative_cancel_num and all(p < 0 for p in profits[-negative_cancel_num:]):
            logging.info(f"Negative profit in last {negative_cancel_num} iterations. Exiting.")
            break

        # Check elapsed time
        if datetime.now() - start_time > timedelta(hours=24):
            logging.info("Running time exceeded 24 hours. Exiting.")
            break

    logging.info(f"Finished. Total profit: {total_profit}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", required=True, help="Path to the YAML config file containing API keys")
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. BTCUSDT")
    parser.add_argument("--quantity", required=True, help="Quantity to operate with")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    setup_logging(f"{args.symbol}_")

    logging.info(f"Script called with: {' '.join(sys.argv)}")
    logging.info(str(args))

    main(args.symbol, float(args.quantity), args.config, args.dry_run)
