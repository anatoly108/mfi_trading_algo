import logging
import os
from datetime import datetime
import sys
import numpy as np

def setup_logging(log_dir=None, file_suffix="", log_to_stdout=True):
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'out')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{file_suffix}{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.txt"
    log_filepath = os.path.join(log_dir, log_filename)

    # Clear existing logging handlers to prevent inheritance issues
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    handlers = [logging.FileHandler(log_filepath)]  # Log to file by default

    if log_to_stdout:
        handlers.append(logging.StreamHandler())  # Log to console if requested

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    # Define a custom exception hook to log uncaught exceptions
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log KeyboardInterrupt to avoid log spam when user interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    # Set the custom exception hook as the global one
    sys.excepthook = log_exception


def get_seconds_for_an_interval(interval):
    if interval.endswith('m'):  # Minute intervals
        interval_seconds = int(interval.replace("m", "")) * 60
    elif interval.endswith('h'):  # Hour intervals
        interval_seconds = int(interval.replace("h", "")) * 3600
    elif interval == '1d':  # 1 day interval
        interval_seconds = 86400
    elif interval == '3d':  # 3 days interval
        interval_seconds = 86400 * 3
    elif interval == '1w':  # 1 week interval
        interval_seconds = 86400 * 7
    elif interval == '1M':  # 1 month interval
        # This is more complex since months vary in length, but you can use an average or approximate value.
        # Here's an average month length in seconds:
        interval_seconds = 86400 * 30.44
    else:
        raise ValueError("Unsupported interval")

    return interval_seconds


def check_if_candles_are_consistent(symbol, candles, interval):
    candles_times = np.array([candle[0] for candle in candles])
    candles_times_diff = np.unique(np.diff(candles_times))
    milliseconds_for_interval = get_seconds_for_an_interval(interval) * 1000
    if len(candles_times_diff) > 1 or np.any(candles_times_diff != milliseconds_for_interval):
        logging.warning(f"{symbol} inconsistent candles intervals: {candles_times_diff}")
        return False
    else:
        return True


def table(data):
    """
    Create a frequency table from categorical data.

    Parameters:
    - data: List or array of categorical data.

    Returns:
    - freq_table: A dictionary where keys are unique values and values are frequencies.
    """
    freq_table = {}
    for item in data:
        if item in freq_table:
            freq_table[item] += 1
        else:
            freq_table[item] = 1
    return freq_table


def retry_decorator(max_retries=3, delay=1, retry_exceptions=()):
    """
    A decorator to retry a function call in case of ConnectionError,
    ReadTimeout, or additional exceptions specified by the user.

    Parameters:
        max_retries (int): Maximum number of retry attempts.
        delay (int): Delay in seconds between retries.
        retry_exceptions (tuple): Additional exception classes to retry on.
    
    Returns:
        The decorated function that retries when specified exceptions occur.
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
                    # If the exception has an attribute 'code' and it equals 429 (Too Many Requests), retry.
                    if hasattr(e, 'code') and e.code == 429:
                        logging.warning(f"{func.__name__} {e.__class__.__name__}: {e}. Too many requests. Retrying... {attempt + 1}/{max_retries}")
                        attempt += 1
                        time.sleep(delay)
                        continue
                    # If the exception is in the additional retry_exceptions tuple, retry.
                    if retry_exceptions and isinstance(e, retry_exceptions):
                        logging.warning(f"{func.__name__} {e.__class__.__name__}: {e}. Retrying... {attempt + 1}/{max_retries}")
                        attempt += 1
                        time.sleep(delay)
                        continue
                    # Otherwise, log the error (with symbol info if present) and re-raise the exception.
                    symbol = ""
                    if "symbol" in kwargs:
                        symbol = f" Symbol: '{kwargs['symbol']}'"
                    logging.error(f"An error occurred: {func.__name__} {e.__class__.__name__}: {e}{symbol}")
                    raise e
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator
