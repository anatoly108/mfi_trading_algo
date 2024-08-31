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
from functions import load_config, setup_logging, get_1m_candles, calculate_mfi, find_extrema, real_time_extrema

def execute_trade(symbol, quantity, action):
    # Replace these with your Binance API credentials
    API_KEY = 'your_api_key'
    API_SECRET = 'your_api_secret'
    client = Client(API_KEY, API_SECRET)
    
    if action == 'buy':
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        logging.info(f"Market Buy Order: {order}")
    elif action == 'sell':
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        logging.info(f"Market Sell Order: {order}")

def main(symbol, quantity):
    setup_logging()
    
    start_time = datetime.now()
    last_profit = 0
    profits = []
    
    # Load initial candles and MFI
    candles = get_1m_candles(symbol)
    mfi = calculate_mfi(candles)

    last_local_minima = 100
    last_local_maxima = 0
    bought = False
    really_new_candles = []
    total_profit = 0

    for i in range(1, len(mfi)):
        mfi_i = mfi[i]
        
        # minima
        if mfi_i < 30 and mfi_i < last_local_minima:
            last_local_minima = mfi_i
    
    while True:
        # Recalculate MFI with the new candle(s)
        mfi = calculate_mfi(candles)
        # We have to account for the case when we get >1 new candle
        mfi_new_from = len(mfi) - len(really_new_candles)
        mfi_new_to = len(mfi)
        # if len(really_new_candles) == 0, then this for loop won't even start
        for i in range(mfi_new_from, mfi_new_to):
            mfi_i = mfi[i]
        
            # minima
            if mfi_i < 30 and mfi_i < last_local_minima:
                last_local_minima = mfi_i

            diff_to_minima = mfi_i - last_local_minima 
            if mfi_i < 30 and mfi_i > last_local_minima and diff_to_minima > 2 and not bought:
                last_local_minima = 100
                bought = True
                # buy signal
                buy_price = float(candles[i][4]) # TODO: replace with real price from the market order
                logging.info(f"Buy signal: price {buy_price}, MFI {mfi_i}")
                break # this will break only from the mfi for loop - we can't sell inside this for loop because we would sell for the same price

            if not bought:
                logging.info(f"Not bought, MFI {mfi_i}")
                continue

            # maxima
            if mfi_i > 70 and mfi_i > last_local_maxima:
                last_local_maxima = mfi_i

            diff_to_maxima = last_local_maxima - mfi_i
            if mfi_i > 70 and mfi_i < last_local_maxima and diff_to_maxima > 2:
                last_local_maxima = 0
                bought = False
                # sell signal
                sell_price = float(candles[i][4])
                logging.info(f"Sell signal: price {sell_price}, MFI {mfi_i}")
                
                profit = (sell_price/buy_price)*100
                total_profit += profit
                logging.info(f"Current trade profit: {profit}%")
                logging.info(f"Total profit: {profit}%")

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

        # Simulate profit calculation (actual implementation will vary)
        # Assuming profit calculation based on buy/sell orders
        # current_profit = 0  # Placeholder for actual profit calculation
        # profits.append(current_profit)
        
        # Check last 3 profits
        # if len(profits) > 3 and all(p <= 0 for p in profits[-3:]):
        #     logging.info("Negative profit in last 3 iterations. Exiting.")
        #     break

        # Check elapsed time
        if datetime.now() - start_time > timedelta(hours=24):
            logging.info("Running time exceeded 24 hours. Exiting.")
            break

if __name__ == "__main__":
    asset = 'BTCUSDT'  # Example asset
    quantity = 0.01    # Example quantity
    main(asset, quantity)