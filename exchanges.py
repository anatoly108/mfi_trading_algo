import yaml
from abc import ABC, abstractmethod
from binance.client import Client as BinanceClient
from pymexc import spot
import os
import logging
import numpy as np

class Exchange(ABC):
    def __init__(self, config_path: str):
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
    def calculate_liquidity_score(self, symbol: str, depth: int):
        pass


class Binance(Exchange):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.client = BinanceClient(self.api_key, self.api_secret)
    
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        candles = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        # time, open, high, low, close, volume
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    def execute_market_order_internal(self, symbol: str, side: str, quantity: float):
        if side.upper() == "BUY":
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
        elif side.upper() == "SELL":
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
        else:
            raise ValueError("Side must be either 'BUY' or 'SELL'")
        
        logging.info(f"Market {side} Order: {order}")

        final_price = np.mean([float(fill['price']) for fill in order['fills']])

        return {'price': final_price, 'order_obj': order}
    
    def get_ticker_data(self, symbol: str):
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url)
        return(response.json())

    def get_all_spot_usdt_pairs(self):
        exchange_info = self.client.get_exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']
        ]
        return usdt_pairs

    def get_taker_fee_fraction(self):
        return 0.075/100

    def get_all_ticker_data(self):
        return self.client.get_ticker(type="MINI")

    def calculate_liquidity_score(self, symbol, depth=200):
        """
        Calculate a liquidity score (0 to 1) based on spread and order book volume for a given symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            depth (int): How deep into the order book to look for volume calculation (default is 5 levels).
        
        Returns:
            float: Liquidity score from 0 to 1, where 1 is highly liquid and 0 is illiquid.
        """
        # Get the order book for the symbol
        order_book = self.client.get_order_book(symbol=symbol, limit=200)

        # Access top bids and asks
        bids = order_book['bids']  # List of [price, quantity]
        asks = order_book['asks']  # List of [price, quantity]

        # Calculate the spread (difference between the best bid and ask)
        highest_bid = float(bids[0][0])
        lowest_ask = float(asks[0][0])
        spread = lowest_ask - highest_bid

        # Calculate total volume on top 'depth' bid and ask levels
        total_bid_volume = sum(float(bid[1]) for bid in bids[:min(len(bids), depth)])
        total_ask_volume = sum(float(ask[1]) for ask in asks[:min(len(asks), depth)])

        # Normalize spread (assuming a larger spread > 0.01 is bad)
        max_spread = 0.01  # Adjust based on the symbol (for very volatile pairs)
        spread_score = max(0, 1 - (spread / max_spread))

        # Normalize volume (higher volume is better, 0 if no volume)
        max_volume = max(total_bid_volume, total_ask_volume)
        if max_volume == 0:
            volume_score = 0  # No liquidity if no volume
        else:
            volume_score = min(total_bid_volume, total_ask_volume) / max_volume

        # Combine spread and volume into a final liquidity score (weighted)
        liquidity_score = 0.5 * spread_score + 0.5 * volume_score

        return liquidity_score

class Mexc(Exchange):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        # Initialize the MEXC client with API key and secret
        self.client = spot.HTTP(api_key=self.api_key, api_secret=self.api_secret)

    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        # Fetch Kline/Candlestick data
        candles = self.client.klines(symbol=symbol, interval=interval, limit=limit, start_time=startTime, end_time=endTime)
        # time, open, high, low, close, volume
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    def execute_market_order_internal(self, symbol: str, side: str, quantity: float):
        # Place a market order
        order = self.client.create_order(symbol=symbol, side=side.upper(), order_type="MARKET", quantity=quantity)
        
        # Get order_id from the response
        order_id = order['orderId']

        # Sleep for a short while to let the order be fully processed
        sleep(1)  # Adjust the sleep duration as needed

        # Fetch the order details using the order_id
        order_info = self.client.get_order(symbol=symbol, order_id=order_id)

        # Calculate final price from the fills (assuming `dealList` contains the execution details)
        if 'dealList' in order_info and order_info['dealList']:
            final_price = np.mean([float(deal['price']) for deal in order_info['dealList']])
        else:
            raise ValueError("No deal information found for the order.")

        logging.info(f"Market {side} Order: {order_info}")

        return {'price': final_price, 'order_obj': order_info}

    def get_ticker_data(self, symbol: str):
        # Get 24-hour ticker data for a given symbol
        ticker = self.client.ticker_24h(symbol)
        return ticker

    def get_all_ticker_data(self):
        tickers = self.client.ticker_24h()
        return tickers

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

    def calculate_liquidity_score(self, symbol: str, depth=5):
        raise Exception("Not implemented")
