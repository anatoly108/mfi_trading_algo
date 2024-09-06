import yaml
from abc import ABC, abstractmethod
from binance.client import Client as BinanceClient
from mexc_sdk import Spot as MexcClient

# Base class defining the interface
class Exchange(ABC):
    def __init__(self, config_path: str):
        # Read the config file and load API keys
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Dynamically load API keys based on the class name (binance/mexc)
        self.api_key = config.get(self.__class__.__name__.lower(), {}).get('api_key')
        self.api_secret = config.get(self.__class__.__name__.lower(), {}).get('api_secret')

    # Abstract methods to be implemented by child classes
    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int):
        pass

    @abstractmethod
    def execute_trade(self, symbol: str, side: str, quantity: float):
        pass

    @abstractmethod
    def get_ticker_data(self, symbol: str):
        pass

    @abstractmethod
    def get_all_spot_usdt_pairs(self):
        pass


# Binance-specific class
class Binance(Exchange):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.client = BinanceClient(self.api_key, self.api_secret)
    
    def get_candles(self, symbol: str, interval: str, limit: int = 100):
        candles = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    def execute_trade(self, symbol: str, side: str, quantity: float):
        if side.upper() == "BUY":
            order = self.client.order_market_buy(symbol=symbol, quantity=quantity)
        elif side.upper() == "SELL":
            order = self.client.order_market_sell(symbol=symbol, quantity=quantity)
        else:
            raise ValueError("Side must be either 'BUY' or 'SELL'")
        return order
    
    def get_ticker_data(self, symbol: str):
        ticker = self.client.get_ticker(symbol=symbol)
        return ticker

    def get_all_spot_usdt_pairs(self):
        exchange_info = self.client.get_exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING' and symbol['isSpotTradingAllowed']
        ]
        return usdt_pairs


# MEXC-specific class
class Mexc(Exchange):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.client = MexcClient(self.api_key, self.api_secret)

    def get_candles(self, symbol: str, interval: str, limit: int = 100):
        interval_mapping = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
        candles = self.client.candles(symbol, interval_mapping[interval], limit)
        formatted_candles = [
            [int(candle['t']), float(candle['o']), float(candle['h']), float(candle['l']), float(candle['c']), float(candle['v'])]
            for candle in candles
        ]
        return formatted_candles

    def execute_trade(self, symbol: str, side: str, quantity: float):
        order = self.client.market_order(symbol, side.upper(), quantity)
        return order

    def get_ticker_data(self, symbol: str):
        ticker = self.client.ticker_24hr(symbol)
        return ticker

    def get_all_spot_usdt_pairs(self):
        exchange_info = self.client.exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['symbol'].endswith('USDT') and symbol['status'] == 'ENABLED' and symbol['isSpotTradingAllowed']
        ]
        return usdt_pairs
