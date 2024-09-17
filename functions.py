import logging
import os

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
