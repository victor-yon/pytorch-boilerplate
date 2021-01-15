import logging
import sys
from pathlib import Path
from typing import Optional, TextIO, Union


class ColorFormatter(logging.Formatter):
    """
    Logging formatter supporting colored output.
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


class SexyLogger(logging.Logger):
    console_handler: logging.Handler = None
    file_handler: logging.Handler = None

    def __init__(self,
                 logger_name: str,
                 console_log_output: Optional[TextIO] = None,
                 console_log_level: Union[int, str] = 0,
                 console_log_color: bool = True,
                 console_log_template: str = '%(asctime)s.%(msecs)03d |%(levelname)-8s| %(message)s',
                 console_log_date: str = '%H:%M:%S',
                 file_enable: bool = False,
                 file_path: Optional[Union[Path, str]] = None,
                 file_log_level: Union[int, str] = 0,
                 file_template: str = '%(asctime)s %(levelname)-8s (%(module)s) %(message)s',
                 file_date_template: str = None):
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
        :param file_enable: If true the log will be also saved in a file (the file_path is then mandatory)
        :param file_path: The path of the file where to save the logs
        :param file_log_level: The level filter of the file log
        :param file_template: The template of the file log messages
                (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
        :param file_date_template: The template of the file log messages' dates
                (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
        """

        super().__init__(logger_name)

        if console_log_output is None:
            console_log_output = sys.stdout  # Default value

        # Create console handler
        # TODO send warnings and errors in stderr
        self.console_handler = logging.StreamHandler(console_log_output)

        # Set console log level (and the global level at the same time)
        self.set_console_level(console_log_level)

        # Create and set formatter, add console handler to logger
        if console_log_color:
            console_formatter = ColorFormatter(fmt=console_log_template, datefmt=console_log_date)
        else:
            console_formatter = logging.Formatter(fmt=console_log_template, datefmt=console_log_date)
        self.console_handler.setFormatter(console_formatter)
        self.addHandler(self.console_handler)

        # If log file is enable, set it up now
        if file_enable:
            self.enable_log_file(file_path, file_log_level, file_template, file_date_template)

    def enable_log_file(self,
                        file_path: Union[Path, str] = None,
                        file_log_level: Union[int, str] = 0,
                        file_template: str = '%(asctime)s %(levelname)-8s (%(module)s) %(message)s',
                        file_date_template: str = None) -> None:
        """
        Enable the log file handle. If already enable nothing happen.

        :param file_path: The path of the file where to save the logs
        :param file_log_level: The level filter of the file log
        :param file_template: The template of the file log messages
                (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
        :param file_date_template: The template of the file log messages' dates
                (see https://docs.python.org/3/library/logging.html#logrecord-attributes)
        """

        # Skip all if already set
        if self.file_handler is not None:
            return

        # Convert and check path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path is None or not file_path.parent.is_dir():
            raise ValueError(f'Invalid log path: {file_path}')

        # Create log file handler
        self.file_handler = logging.FileHandler(file_path)

        # Create and set formatter, add log file handler to logger
        self.file_handler.setFormatter(logging.Formatter(fmt=file_template, datefmt=file_date_template))
        self.set_file_level(file_log_level)
        self.addHandler(self.file_handler)

    def set_console_level(self, level: Union[int, str]) -> None:
        """
        Set the console log minimum level to show.
        Also change the global level to the minimum between the two handler.

        :param level: The log level
        """

        # Set the handler level
        self.console_handler.setLevel(level)

        if self.file_handler is None:
            # Set the global log level at the same point because it is the only handler
            self.setLevel(level)
        else:
            # Set global log level to the minimum value between the two handler
            self.setLevel(min(self.console_handler.level, self.file_handler.level))

    def set_file_level(self, level: Union[int, str]) -> None:
        """
        Set the file log minimum level to save.
        Also change the global level to the minimum between the two handler.

        :param level: The log level
        """

        # Set the handler level
        self.file_handler.setLevel(level)
        # Set global log level to the minimum value between the two handler
        self.setLevel(min(self.console_handler.level, self.file_handler.level))


# Create the logger singleton
logger: SexyLogger = SexyLogger(logger_name='pytorch-boilerplate')
