import yaml
from abc import ABC, abstractmethod
from binance.client import Client as BinanceClient
from pymexc import spot
import os
import logging
import numpy as np
import requests
import time
from contextlib import nullcontext

def retry_decorator(max_retries=3, delay=1):
    """
    A decorator to retry a function call in case of ConnectionError.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
                    logging.warning(f"{func.__name__} {e.__class__.__name__}: {e}. Retrying... {attempt + 1}/{max_retries}")
                    attempt += 1
                    time.sleep(delay)
                except Exception as e:
                    if hasattr(e, 'code') and e.code == 429:  # Too Many Requests error code
                        logging.warning(f"{func.__name__} {e.__class__.__name__}: {e}. Too many requests. Retrying... {attempt + 1}/{max_retries}")
                        attempt += 1
                        time.sleep(delay)
                        continue
                        
                    logging.error(f"An error occurred: {func.__name__} {e.__class__.__name__}: {e}")
                    raise e

            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator

def semaphore_decorator():
    """
    A decorator to retry a function call in case of ConnectionError,
    with access to the class instance (self).
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Use the provided semaphore if it's there, otherwise no-op (nullcontext)
            with self.semaphore if self.semaphore else nullcontext():
                return func(self, *args, **kwargs)
        
        return wrapper
    return decorator

# Metaclass to automatically apply the decorator
class RetryMeta(type):
    def __new__(cls, name, bases, dct):
        for attr, value in dct.items():
            if callable(value) and not attr.startswith('__'):
                dct[attr] = retry_decorator()(value)
        return super().__new__(cls, name, bases, dct)

class Exchange(metaclass=RetryMeta):
    def __init__(self, config_path, semaphore):
        self.semaphore = semaphore

        if not os.path.exists(config_path):
            self.api_key = None
            self.api_secret = None
            return

        # Read the config file and load API keys
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Dynamically load API keys based on the class name (binance/mexc)
        self.api_key = config.get(self.__class__.__name__.lower(), {}).get('api_key')
        self.api_secret = config.get(self.__class__.__name__.lower(), {}).get('api_secret')
        
    def execute_market_order(self, symbol: str, side: str, quantity: float, dry_run: bool):
        if dry_run:
            logging.info(f"Dry run {side}")
            return {'price': None}

        self.execute_market_order_internal(symbol, side, quantity)

    # Abstract methods to be implemented by child classes
    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        pass

    @abstractmethod
    def execute_market_order_internal(self, symbol: str, side: str, quantity: float):
        pass

    @abstractmethod
    def get_ticker_data(self, symbol: str):
        pass

    @abstractmethod
    def get_all_ticker_data(self):
        pass

    @abstractmethod
    def get_all_spot_usdt_pairs(self):
        pass

    @abstractmethod
    def get_taker_fee_fraction(self):
        pass

    @abstractmethod
    def get_order_book(self, symbol, limit=100):
        pass

class Binance(Exchange):
    def __init__(self, config_path="", semaphore=None):
        super().__init__(config_path, semaphore)
    
    @semaphore_decorator()
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        candles = BinanceClient().get_klines(symbol=symbol, interval=interval, limit=limit, startTime=startTime, endTime=endTime)
        # time, open, high, low, close, volume
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    @semaphore_decorator()
    def execute_market_order_internal(self, symbol: str, side: str, quantity: float):
        if side.upper() == "BUY":
            order = BinanceClient(self.api_key, self.api_secret).order_market_buy(symbol=symbol, quantity=quantity)
        elif side.upper() == "SELL":
            order = BinanceClient(self.api_key, self.api_secret).order_market_sell(symbol=symbol, quantity=quantity)
        else:
            raise ValueError("Side must be either 'BUY' or 'SELL'")
        
        logging.info(f"Market {side} Order: {order}")

        final_price = np.mean([float(fill['price']) for fill in order['fills']])

        return {'price': final_price, 'order_obj': order}
    
    @semaphore_decorator()
    def get_ticker_data(self, symbol: str):
        return(BinanceClient().get_ticker(symbol=symbol, type="MINI"))

    def get_all_spot_usdt_pairs(self):
        exchange_info = BinanceClient().get_exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']
        ]
        return usdt_pairs

    def get_taker_fee_fraction(self):
        return 0.075/100

    @semaphore_decorator()
    def get_all_ticker_data(self):
        return BinanceClient().get_ticker(type="MINI")

    @semaphore_decorator()
    def get_order_book(self, symbol, limit=100):
        return BinanceClient().get_order_book(symbol=symbol, limit=limit)

class Mexc(Exchange):
    def __init__(self, config_path="", semaphore=None):
        super().__init__(config_path, semaphore)
        # Initialize the MEXC client with API key and secret
        self.client = spot.HTTP(api_key=self.api_key, api_secret=self.api_secret)

    @semaphore_decorator()
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        # Fetch Kline/Candlestick data
        candles = self.client.klines(symbol=symbol, interval=interval, limit=limit, start_time=startTime, end_time=endTime)
        # time, open, high, low, close, volume
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    @semaphore_decorator()
    def execute_market_order_internal(self, symbol: str, side: str, quantity: float):
        # Place a market order
        order = self.client.new_order(symbol=symbol, side=side.upper(), order_type="MARKET", quantity=quantity)
        
        # Get order_id from the response
        order_id = order['orderId']

        # Sleep for a short while to let the order be fully processed
        time.sleep(1)  # Adjust the sleep duration as needed

        # Fetch the order details using the order_id
        order_info = self.client.query_order(symbol=symbol, order_id=order_id)

        max_tries = 3
        tries = 0
        while order_info["status"] != "FILLED":
            logging.info(f"Market {side} order not filled yet; waiting, current tries: {tries}")
            if tries == max_tries:
                raise Exception("Market order not filled.")
            
            time.sleep(1)
            order_info = self.client.query_order(symbol=symbol, order_id=order_id)
            tries += 1

        return order_info
        # final_price "price"

        # Calculate final price from the fills (assuming `dealList` contains the execution details)
        if 'dealList' in order_info and order_info['dealList']:
            final_price = np.mean([float(deal['price']) for deal in order_info['dealList']])
        else:
            raise ValueError("No deal information found for the order.")

        logging.info(f"Market {side} Order: {order_info}")

        return {'price': final_price, 'order_obj': order_info}

    @semaphore_decorator()
    def get_ticker_data(self, symbol: str):
        # Get 24-hour ticker data for a given symbol
        ticker = self.client.ticker_24h(symbol)
        return ticker

    @semaphore_decorator()
    def get_all_ticker_data(self):
        tickers = self.client.ticker_24h()
        return tickers

    @semaphore_decorator()
    def get_all_spot_usdt_pairs(self):
        # Fetch all trading pairs and filter for USDT pairs
        exchange_info = self.client.exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == "USDT" and "SPOT" in symbol['permissions']
        ]
        return usdt_pairs
    
    def get_taker_fee_fraction(self):
        return 0.02/100

    @semaphore_decorator()
    def get_order_book(self, symbol, limit=100):
        return self.client.order_book(symbol=symbol, limit=limit)
