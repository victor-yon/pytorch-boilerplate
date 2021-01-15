import logging
import sys
from pathlib import Path
from typing import Optional, TextIO, Union


class ColorFormatter(logging.Formatter):
    """
    Logging formatter supporting colored output
    """

    # TODO check color working on windows (see https://stackoverflow.com/a/61043789/2666094)
    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[0;33m",  # bright/bold yellow
        logging.INFO: "",  # default color
        logging.DEBUG: "\033[0;37m"  # white / light gray
    }

    RESET_CODE = "\033[0m"

    def format(self, record):
        return self.COLOR_CODES[record.levelno] + super().format(record) + self.RESET_CODE


def setup_logger(logger_name: str,
                 console_log_output: Optional[TextIO] = None,
                 console_log_level: int = 0,
                 console_log_color: bool = True,
                 console_log_template: str = '%(asctime)s.%(msecs)03d |%(levelname)-8s| %(message)s',
                 console_log_date: str = '%H:%M:%S',
                 log_file_enable: bool = False,
                 logfile_path: Optional[Union[Path, str]] = None,
                 logfile_log_level: int = 0,
                 logfile_template: str = '%(asctime)s %(levelname)-8s (%(module)s) %(message)s',
                 logfile_date: str = None) -> logging.Logger:
    """
    Setup a logger.

    :param logger_name: The name of the logger for this project
    :param console_log_output: The output channel of the console log (default stdout)
    :param console_log_level: The level filter of the console log
    :param console_log_color: If true the console log will be colorized
    :param console_log_template: The template of the console log messages
            (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
    :param console_log_date: The template of the console log messages' dates
            (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
    :param log_file_enable: If true the log will be also saved in a file (the logfile_path is then mandatory)
    :param logfile_path: The path of the file where to save the logs
    :param logfile_log_level: The level filter of the file log
    :param logfile_template: The template of the file log messages
            (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
    :param logfile_date: The template of the file log messages' dates
            (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
    :return: A new logger instance.
    """

    if console_log_output is None:
        console_log_output = sys.stdout

    # Create logger
    new_logger = logging.getLogger(logger_name)

    if log_file_enable:
        # Set global log level to the minimum value between the two handler
        new_logger.setLevel(min(console_log_level, logfile_log_level))
    else:
        # Set global to the console value because file is disable
        new_logger.setLevel(console_log_level)

    # Create console handler
    console_handler = logging.StreamHandler(console_log_output)

    # Set console log level
    console_handler.setLevel(console_log_level)

    # Create and set formatter, add console handler to logger
    if console_log_color:
        console_formatter = ColorFormatter(fmt=console_log_template, datefmt=console_log_date)
    else:
        console_formatter = logging.Formatter(fmt=console_log_template, datefmt=console_log_date)
    console_handler.setFormatter(console_formatter)
    new_logger.addHandler(console_handler)

    if log_file_enable:
        if isinstance(logfile_path, str):
            logfile_path = Path(logfile_path)

        if logfile_path is None or not logfile_path.parent.is_dir():
            raise ValueError(f'Invalid log path: {logfile_path}')

        # Create log file handler
        logfile_handler = logging.FileHandler(logfile_path)

        # Set log file log level
        logfile_handler.setLevel(logfile_log_level)

        # Create and set formatter, add log file handler to logger
        logfile_handler.setFormatter(logging.Formatter(fmt=logfile_template, datefmt=logfile_date))
        new_logger.addHandler(logfile_handler)

    return new_logger


# Create the logger singleton
logger = setup_logger(logger_name='pytorch-boilerplate',
                      console_log_level=logging.INFO,
                      log_file_enable=True,
                      logfile_path=Path("./", "run.log"),
                      logfile_log_level=logging.DEBUG)
