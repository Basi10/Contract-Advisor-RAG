import logging


class Logger:
    """Logger class for logging messages to a file."""

    def __init__(self, file_name: str, basic_level=logging.INFO):
        """Initilize logger class with file name to be written and default log level.

        Args:
            file_name (str): _description_
            basic_level (_type_, optional): _description_. Defaults to logging.INFO.
        """
        # Gets or creates a logger
        logger = logging.getLogger(__name__)

        # set log level
        logger.setLevel(basic_level)

        # define file handler and set formatter

        file_handler = logging.FileHandler(f'../logs/{file_name}')
        formatter = logging.Formatter(
            '%(asctime)s : %(levelname)s : %(name)s : %(message)s', '%m-%d-%Y %H:%M:%S')

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def get_app_logger(self) -> logging.Logger:
        """Return the logger object.

        Returns:
            logging.Logger: logger object.
        """
        return self.logger