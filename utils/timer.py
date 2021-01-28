import time
from math import floor

from codetiming import Timer, TimerError

from utils.logger import logger


class SectionTimer(Timer):
    def __init__(self, section_name: str):
        super().__init__(name=section_name)
        self.logger = logger.info

    def start(self) -> None:
        self.logger(f'Start {self.name}...')
        super().start()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        self.last = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(f'Completed {self.name} in {duration_to_str(self.last)}')
        if self.name:
            self.timers.add(self.name, self.last)

        return self.last


def duration_to_str(sec: float, precision: int = 2):
    """
    Transform a duration (in sec) into a human readable string.

    :param sec: The number of second of the duration. Decimals are milliseconds.
    :param precision: The number of unit we want. If 0 print all units.
    :return: A human readable representation of the duration.
    """

    assert sec >= 0, 'Negative duration not supported'

    # Null duration
    if sec == 0:
        return "0ms"

    # Convert to ms
    mills = floor(sec * 1_000)

    # Less than 1 millisecond
    if mills == 0:
        return "<1ms"

    periods = [
        ('d', 1_000 * 60 * 60 * 24),
        ('h', 1_000 * 60 * 60),
        ('m', 1_000 * 60),
        ('s', 1_000),
        ('ms', 1)
    ]

    strings = []
    for period_name, period_mills in periods:
        if mills >= period_mills:
            period_value, mills = divmod(mills, period_mills)
            strings.append(f"{period_value}{period_name}")

    if precision:
        strings = strings[:precision]

    return " ".join(strings)