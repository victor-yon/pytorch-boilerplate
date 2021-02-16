import time
from dataclasses import dataclass
from random import random
from typing import Any, Optional

from utils.timer import duration_to_str


@dataclass
class ProgressBarMetrics:
    name: str
    last_value: Optional[float] = None
    last_printed_value: Optional[float] = None
    print_type: str = 'f'  # Accept f (float) or %
    more_is_good: bool = True
    _printed: bool = False

    def update(self, value: float) -> None:
        """
        Update the value and keep track of the previous one if it was already printed.
        :param value: The new value for this metric.
        """
        if value is not None:
            # Keep the last value until then so we can always show the good indicator
            if self._printed:
                self.last_printed_value = self.last_value
            self._printed = False
            self.last_value = value

    def printed(self) -> None:
        """ Register than the current value was printed (useful for indicators) """
        self._printed = True

    def evolution_indicator_str(self) -> str:
        """
        Return a colored character depending of the direction of the value.
        """
        # TODO global variable for colors
        good_color = '\033[0;92m'  # Green text
        bad_color = '\033[0;91m'  # Red text
        no_evolution_color = '\033[0;33m'  # Orange text
        reset_color = '\033[0m'

        # Case with None previous value
        if self.last_printed_value is None:
            return ' '

        loss_diff = self.last_value - self.last_printed_value

        if loss_diff > 0:
            return f'{good_color if self.more_is_good else bad_color}▲{reset_color}'
        elif loss_diff < 0:
            return f'{bad_color if self.more_is_good else good_color}▼{reset_color}'
        else:
            return f'{no_evolution_color}={reset_color}'

    def __str__(self) -> str:
        # Case with None value
        if self.last_value is None:
            return f'{self.name}: None   '

        return f'{self.name}:{self.evolution_indicator_str()}' + (
            f'{self.last_value or 0:<7.2%}' if self.print_type == '%' else f'{self.last_value or 0:7.5f}')


class ProgressBar:
    def __init__(self, nb_epoch_batch: int, nb_epoch: int, task_name: str = 'progress', length: int = 50,
                 epoch_char: str = '⎼', fill_char: str = ' ', refresh_time: int = 500, auto_display: bool = True,
                 with_accuracy: bool = False):
        """
        Create a machine learning progress bar to visual print and tracking progress.

        :param nb_epoch_batch: The number of batch to process per epoch.
        :param nb_epoch: The number of epoch.
        :param task_name: The name of the task.
        :param length: The size of the visual progress bar (number of characters)
        :param epoch_char: The character used for epoch progress done.
        :param fill_char: The character used for epoch progress pending.
        :param refresh_time: The minimal time delta between two auto print, 0 for all auto print (in milliseconds).
        :param auto_display: If true the bar will be automatically printed at the start, the end and after every value
        update if the minimal refresh time allow it.
        """
        self.nb_epoch_batch = nb_epoch_batch
        self.current_batch = 0

        self.nb_epoch = nb_epoch
        self.current_epoch = 0

        self.task_name = task_name
        self.length = length
        self.epoch_char = epoch_char
        self.fill_char = fill_char

        self._start_time = None
        self._end_time = None
        self._last_print = None
        self._refresh_time = refresh_time
        self._auto_display = auto_display

        self.loss = ProgressBarMetrics('loss', more_is_good=False)
        self.with_accuracy = with_accuracy
        self.accuracy = ProgressBarMetrics('accuracy', print_type='%')

    def incr_batch(self, loss: Optional[float] = None, accuracy: Optional[float] = None) -> None:
        """
        Increase by one the number of batch.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        :param loss: The loss for this batch
        :param accuracy: The accuracy for this loss
        """
        self.loss.update(loss)
        self.accuracy.update(accuracy)
        self.current_batch += 1
        if self._auto_display:
            self.lazy_print()

    def incr_epoch(self) -> None:
        """
        Increase by one the number of epoch.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        """
        self.current_epoch += 1
        if self._auto_display:
            self.lazy_print()

    def get_eta(self) -> float:
        """
        Get the "estimated time of arrival" to reach 100% of this task.
        :return: The estimated value in second.
        """
        delta_t = time.perf_counter() - self._start_time
        progress = self.get_task_progress() * 100
        # Deal with not started task
        if delta_t == 0 or progress == 0:
            return float('inf')
        # Max is use to avoid negative value at the end of the task
        return max(((100 * delta_t) / progress) - delta_t, 0)

    def get_task_progress(self) -> float:
        """
        :return: The completed percentage of this task.
        """
        return self.current_batch / (self.nb_epoch * self.nb_epoch_batch)

    def get_epoch_progress(self) -> float:
        """
        :return: The completed percentage of the current epoch.
        """
        progress = (self.current_batch % self.nb_epoch_batch) / self.nb_epoch_batch
        # Keep 0% for the start but change to 100% for the end of each sub tasks
        if progress == 0 and self.current_epoch != 0:
            return 1.0
        return progress

    def print(self) -> None:
        """ Force print the progression. """
        print(f'{self}', end='\r', flush=True)
        self._last_print = time.perf_counter()
        self.loss.printed()
        self.accuracy.printed()

    def lazy_print(self) -> None:
        """ Print the bar if the minimal refresh time allow it. """
        if self._last_print is None or self._refresh_time == 0 or \
                (time.perf_counter() - self._last_print) * 1_000 >= self._refresh_time:
            self.print()

    def __str__(self) -> str:
        """
        :return: The progress bar formatted as a string.
        """
        # Epoch progress
        epoch_progress = self.get_epoch_progress()
        nb_fill_char = int(epoch_progress * self.length)
        bar = self.epoch_char * nb_fill_char + self.fill_char * (self.length - nb_fill_char)

        # Global task
        task_progress = self.get_task_progress()
        nb_task_char = int(task_progress * self.length)
        # Gray background for task loading
        bar = '\033[0;100m' + bar[:nb_task_char] + '\033[0m' + bar[nb_task_char:]

        string = f'{self.task_name}⎹ {task_progress:7.2%}⎹{bar}⎸' \
                 f'ep. {self.current_epoch}/{self.nb_epoch} {epoch_progress:<4.0%}' \
                 f'⎹ {self.loss}'

        if self.with_accuracy:
            string += f'⎹ {self.accuracy}'

        if self._end_time is None:
            # Still in progress
            eta = duration_to_str(self.get_eta(), precision='s')
            string += f'⎹ ETA: {eta}'
        else:
            # Task ended or interrupted
            duration = duration_to_str(self._end_time - self._start_time, precision='s')
            string += f'⎹ {duration}'

        return string

    def start(self):
        """
        Start timer and display the bar if auto display enable.
        """
        assert self._start_time is None, 'The progress bar can\'t be started twice.'
        self._start_time = time.perf_counter()
        if self._auto_display:
            self.print()

    def __enter__(self) -> "ProgressBar":
        """
        Start timer and display the bar if auto display enable.
        :return: The current progress bar object.
        """
        self.start()
        return self

    def stop(self):
        """
        Save end time and display the bar if auto display enable.
        """
        assert self._end_time is None, 'The progress bar can\'t be stopped twice.'
        self._end_time = time.perf_counter()
        if self._auto_display:
            self.print()
            # TODO print a summary for the last one
            print()  # New line at the end

    def __exit__(self, *exc_info: Any) -> None:
        """
        Final display of the bar if auto display enable.
        """
        self.stop()


if __name__ == '__main__':
    nb_batch = 10
    nb_epoch = 2

    with ProgressBar(nb_batch, nb_epoch, refresh_time=200, with_accuracy=True) as p:
        for epoch_i in range(nb_epoch):
            p.incr_epoch()
            for batch in range(nb_batch):
                # Do stuff...
                time.sleep(1)
                p.incr_batch(loss=batch, accuracy=random())

    print('end')
