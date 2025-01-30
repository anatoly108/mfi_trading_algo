import yaml
from abc import ABC, abstractmethod
from binance.client import Client as BinanceClient
import binance
from pymexc import spot
import os
import logging
import numpy as np
import requests
import time
import hashlib
import hmac
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
                    
                    symbol = ""
                    if "symbol" in kwargs.keys():
                        symbol = f" Symbol: '{kwargs['symbol']}'"
                    logging.error(f"An error occurred: {func.__name__} {e.__class__.__name__}: {e}{symbol}")
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

        return self.execute_market_order_internal(symbol, side, quantity)

    # Abstract methods to be implemented by child classes
    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int, market="spot"):
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

    @abstractmethod
    def get_all_perp_usdt_pairs(self):
        pass

    @abstractmethod
    def get_open_interest(self, symbol, interval, startTime=None, endTime=None):
        pass

    @abstractmethod
    def get_funding_rate(self, symbol):
        pass

class Binance(Exchange):
    def __init__(self, config_path="", semaphore=None):
        super().__init__(config_path, semaphore)
    
    @semaphore_decorator()
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int, market="spot"):
        if market == "spot":
            candles = BinanceClient().get_klines(symbol=symbol, interval=interval, limit=limit, startTime=startTime, endTime=endTime)
        elif market == "futures":
            candles = BinanceClient().futures_klines(symbol=symbol, interval=interval, limit=limit, startTime=startTime, endTime=endTime)
        else:
            raise Exception(f"Unknown market: {market}")

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
        usdt_pairs = []
        usdc_pairs = []

        # Separate USDT and USDC pairs
        for symbol in exchange_info['symbols']:
            if symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']:
                if symbol['symbol'].endswith('USDT'):
                    usdt_pairs.append(symbol['symbol'])
                elif symbol['symbol'].endswith('USDC'):
                    usdc_pairs.append(symbol['symbol'])

        # Now filter out USDC pairs that have a corresponding USDT pair
        final_pairs = usdt_pairs + [usdc for usdc in usdc_pairs if usdc.replace('USDC', 'USDT') not in usdt_pairs]

        return final_pairs

    def get_taker_fee_fraction(self):
        return 0.075/100

    @semaphore_decorator()
    def get_all_ticker_data(self):
        return BinanceClient().get_ticker(type="MINI")

    @semaphore_decorator()
    def get_order_book(self, symbol, limit=100):
        return BinanceClient().get_order_book(symbol=symbol, limit=limit)

    @semaphore_decorator()
    def get_all_perp_usdt_pairs(self):
        # Fetch all futures symbols
        futures_info = BinanceClient().futures_exchange_info()
        
        # Filter for PERPETUAL contract types
        perpetual_symbols = [symbol['symbol'] for symbol in futures_info['symbols'] if symbol['contractType'] == 'PERPETUAL' and 'USD' in symbol['quoteAsset']]
        
        return perpetual_symbols

    @semaphore_decorator()
    def get_open_interest(self, symbol, interval, startTime=None, endTime=None):
        oi_data = []

        startTimeMillis = None
        endTimeMillis = None

        if startTime is not None and endTime is not None:
            # Convert datetime to milliseconds for Binance API
            startTimeMillis = int(startTime.timestamp() * 1000)
            endTimeMillis = int(endTime.timestamp() * 1000)

        # Fetch OI historical data
        # https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics
        oi_response = BinanceClient().futures_open_interest_hist(
            symbol=symbol,
            period=interval,
            limit=500,
            startTime=startTimeMillis,
            endTime=endTimeMillis
        )

        # Extract Open Interest values from the response
        for data_point in oi_response:
            # in BTCUSD sumOpenInterest is USD
            oi_data.append(float(data_point['sumOpenInterest']))

        return oi_data

    def get_funding_rate(self, symbol):
        # https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1
        response = BinanceClient().futures_funding_rate(
            symbol=symbol,
            limit=1
        )
        if len(response) == 0:
            return None

        return float(response[0]["fundingRate"]) * 100

class Mexc(Exchange):
    def __init__(self, config_path="", semaphore=None):
        super().__init__(config_path, semaphore)
        # Initialize the MEXC client with API key and secret
        self.client = spot.HTTP(api_key=self.api_key, api_secret=self.api_secret)

    @semaphore_decorator()
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int, market="spot"):
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
        logging.info(f"Market {side} order id {order_id}")

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

        # order_info has "price", but it's just wrong: we calculate price from 
        # "cummulativeQuoteQty" which final amounts in quote currency
        quote_quantity = float(order_info["cummulativeQuoteQty"])
        final_price = quote_quantity / quantity
        logging.info(f"Market {side} order info: {order_info}")

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

        # Define the URL
        url = 'https://www.mexc.com/open/api/v2/market/api_symbols'

        # Generate current timestamp in milliseconds
        req_time = str(int(time.time() * 1000))

        # Create the signature string: accessKey + req_time
        message = self.api_key + req_time

        # Create the signature using HMAC-SHA256
        signature = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()

        # Prepare headers for the request
        headers = {
            'ApiKey': self.api_key,
            'Request-Time': req_time,
            'Signature': signature,
            'Content-Type': 'application/json'
        }

        # Send the authenticated request
        response = requests.get(url, headers=headers)

        # Check the response
        if response.status_code == 200:
            symbols = response.json()["data"]["symbol"]
            symbols = [symbol.replace("_", "") for symbol in symbols]

            # Separate USDT and USDC pairs
            usdt_symbols = [symbol for symbol in symbols if symbol.endswith("USDT")]
            usdc_symbols = [symbol for symbol in symbols if symbol.endswith("USDC")]

            # Keep USDC only if the corresponding USDT doesn't exist
            final_symbols = usdt_symbols + [usdc for usdc in usdc_symbols if usdc.replace("USDC", "USDT") not in usdt_symbols]

            return(final_symbols)
        else:
            res_text = ""
            try:
                res_rext = str(response.json())
            except Exception:
                pass

            raise Exception(f"Failed to fetch trading pairs. Status code: {response.status_code}, {res_text}")
    
    def get_taker_fee_fraction(self):
        return 0.02/100

    @semaphore_decorator()
    def get_order_book(self, symbol, limit=100):
        return self.client.order_book(symbol=symbol, limit=limit)
