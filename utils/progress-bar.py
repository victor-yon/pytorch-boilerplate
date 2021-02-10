# Print iterations progress
import time
from typing import Any

from utils.timer import duration_to_str


class ProgressBar:
    def __init__(self, nb_epoch_batch: int, nb_epoch: int, task_name: str = 'progress', length: int = 50,
                 epoch_char: str = '-', fill_char: str = ' ', refresh_time: int = 500, auto_display: bool = True):
        """
        Create a machine learning progress bar to visual print and tracking progress.

        :param nb_epoch_batch: The number of batch to process per epoch.
        :param nb_epoch: The number of epoch.
        :param task_name: The name of the task.
        :param length: The size of the visual progress bar (number of characters)
        :param epoch_char: The character used for epoch progress done.
        :param fill_char: The character used for epoch progress pending.
        :param refresh_time: The minimal time distance between two auto print (in milliseconds).
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

        self._start_time = time.perf_counter()
        self._last_print = time.perf_counter()
        self._refresh_time = refresh_time
        self._auto_display = auto_display

    def incr_batch(self) -> None:
        """
        Increase by one the number of batch.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        """
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
        print('current_epoch:', self.current_epoch)
        print(
            f'current_batch: {self.current_batch} % nb_epoch_batch: {self.nb_epoch_batch} / nb_epoch_batch: {self.nb_epoch_batch}')
        print('progress:', progress)
        if progress == 0 and self.current_epoch != 0:
            return 1.0
        return progress

    def print(self) -> None:
        """ Force print the progression. """
        print(f'\r{self}', end='\r', flush=True)
        self._last_print = time.perf_counter()

    def lazy_print(self) -> None:
        """ Print the bar the minimal refresh time allow it. """
        if (time.perf_counter() - self._last_print) * 1_000 >= self._refresh_time:
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

        eta = duration_to_str(self.get_eta(), precision='s')

        return f'{self.task_name} | {task_progress:7.2%} |{bar}| ' \
               f'epoch {self.current_epoch}/{self.nb_epoch} - {epoch_progress:<6.1%}| ETA: {eta}'

    def __enter__(self) -> "ProgressBar":
        """
        Initial display of the bar if auto display enable.
        :return: The current progress bar object.
        """
        if self._auto_display:
            self.print()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """
        Final display of the bar if auto display enable.
        """
        if self._auto_display:
            self.print()


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


if __name__ == '__main__':
    nb_batch = 10
    nb_epoch = 2

    with ProgressBar(nb_batch, nb_epoch, refresh_time=1) as p:
        for epoch_i in range(nb_epoch):
            p.incr_epoch()
            for batch in range(nb_batch):
                p.incr_batch()
                # Do stuff...
                time.sleep(1)
