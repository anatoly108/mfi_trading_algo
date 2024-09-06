import yaml
from abc import ABC, abstractmethod
from binance.client import Client as BinanceClient
from pymexc import spot
import os

# Base class defining the interface
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

    # Abstract methods to be implemented by child classes
    @abstractmethod
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        pass

    @abstractmethod
    def execute_market_order(self, symbol: str, side: str, quantity: float):
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
    
    def get_candles(self, symbol: str, interval: str, limit: int, startTime: int, endTime: int):
        candles = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        # time, open, high, low, close, volume
        formatted_candles = [
            [int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])]
            for candle in candles
        ]
        return formatted_candles

    def execute_market_order(self, symbol: str, side: str, quantity: float):
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


# MEXC-specific class
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

    def execute_market_order(self, symbol: str, side: str, quantity: float):
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

    def get_all_spot_usdt_pairs(self):
        # Fetch all trading pairs and filter for USDT pairs
        exchange_info = self.client.exchange_info()
        usdt_pairs = [
            symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['quoteAsset'] == "USDT" and "SPOT" in symbol['permissions']
        ]
        return usdt_pairs
