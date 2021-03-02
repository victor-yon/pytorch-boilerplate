import math
from typing import Any, Collection, Iterator, List, Optional

from utils.settings import settings


class BasePlanner:
    """
    Abstract class Base Planner.
    Use one of its children class.
    """

    def __init__(self, runs_basename: Optional[str] = None):
        """
        Create an abstract setting planner object.

        :param runs_basename: The basename of the run names generate by this planner. If defined, the names will be as:
        "basename-num" (eg: "plop-001", ""plop-002", "plop-003", ...). If this parameter is None or empty, the name will
        be generated according to the setting name, which is very descriptive but could be quite long.
        """
        self.runs_basename = runs_basename
        self.num_count = 0

    def __iter__(self) -> Iterator:
        """
        Create the planner iterator.
        :return: Itself
        """
        raise NotImplemented('Iteration abstract method need to be override')

    # noinspection PyTypeChecker
    def __next__(self) -> str:
        """
        Change the next setting in-place, according to the planner
        :return: The name of the current run
        """
        self.num_count += 1

    def __len__(self) -> int:
        """
        Get the length of the planner (the total number of iterations)
        :return: The length
        """
        raise NotImplemented('Length abstract method need to be override')

    def basename(self):
        """
        Format the name of this run according to the count number and the basename defined by the user.
        :return: The basename of this run
        """
        return f'{self.runs_basename}-{self.num_count:03d}'

    def reset_original_values(self) -> None:
        """
        Reset the setting value as it was before the start of the iteration.
        """
        raise NotImplemented('Reset abstract method need to be override ')


class Planner(BasePlanner):
    """
    Simple planner use to start a set of runs with a different values for one setting.
    """

    def __init__(self, setting_name: str, setting_values: Collection, runs_basename: str = ''):
        """
        Create a simple planner that will iterate value of a specific setting.

        :param setting_name: The name of the setting to change.
        :param setting_values: The collection of value to iterate (the collection should be iterable and have a length).
        :param runs_basename: The basename of the run names generate by this planner. If defined, the names will be as:
        "basename-num" (eg: "plop-001", ""plop-002", "plop-003", ...). If this parameter is None or empty, the name will
        be generated according to the setting name, which is very descriptive but could be quite long.
        """
        super().__init__(runs_basename)

        self.setting_name = setting_name
        self.setting_values = setting_values
        self._setting_original_value = None  # Will be set when the iteration start
        self._values_iterator = None

    def __iter__(self) -> Iterator:
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        self._setting_original_value = getattr(settings, self.setting_name)
        self._values_iterator = iter(self.setting_values)
        return self

    def __next__(self) -> str:
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        super().__next__()
        # Get new value
        value = next(self._values_iterator)

        # Set new value
        setattr(settings, self.setting_name, value)

        # Return the name of this run
        return self.format_name(value)

    def __len__(self) -> int:
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return len(self.setting_values)

    def format_name(self, value: Any) -> str:
        """
        Format the name in case the basename is not set.
        As: "setting_name-value" (eg: "epoch-15")

        :param value: The current setting value
        :return: The formatted name
        """
        # If no runs basename provided use the variable name and value as run name
        return self.basename() if self.runs_basename else f'{self.setting_name}-{value}'

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        if self._values_iterator is not None and getattr(settings, self.setting_name) != self._setting_original_value:
            setattr(settings, self.setting_name, self._setting_original_value)


class SequencePlanner(BasePlanner):
    """
    To organise planners by sequential order.
    When the current planner is over the next one on the list will start.
    The total length of this sequence will be the sum of each sub-planners.
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        """
        Create a planner that will iterate some sub-planners one by one.
        The settings will be reset between each sub-planners.

        :param planners: The list of sub-planners to iterate, it can be any subclass of BasePlanner.
        :param runs_basename: The basename of the run names generate by this planner. If defined, the names will be as:
        "basename-num" (eg: "plop-001", ""plop-002", "plop-003", ...). If this parameter is None or empty, the name will
        be generated according to the setting name, which is very descriptive but could be quite long.
        """
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for sequence planner')

        self.planners = planners
        self._current_planner_id = 0
        self._planners_iterator = None
        self._current_planner_iterator = None

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # First iterate over every planners
        self._planners_iterator = iter(self.planners)
        self._current_planner_id = 0
        # Then iterate inside each planner
        self._current_planner_iterator = iter(next(self._planners_iterator))

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        super().__next__()
        try:
            # Try to iterate inside the current planner
            sub_run_name = next(self._current_planner_iterator)
            return self.format_name(sub_run_name)
        except StopIteration:
            # Reset settings as original value before next planner
            self.planners[self._current_planner_id].reset_original_values()
            self._current_planner_id += 1

            # If current planner is over, open the next one
            # If it's already the last planner then the StopIteration will be raise again here
            self._current_planner_iterator = iter(next(self._planners_iterator))

            # Recursive call with the new planner
            return next(self)

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return sum(map(len, self.planners))

    def format_name(self, sub_run_name: str) -> str:
        """
        Format the name in case the basename is not set.
        Use the name of the current sub-planners formatted names

        :param sub_run_name: The name of the current sub-planners names
        :return: The formatted name
        """
        # If no runs basename provided use the name of the last sub-planner
        return self.basename() if self.runs_basename else sub_run_name

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()


class ParallelPlanner(BasePlanner):
    """
    To organise planners in parallel.
    All planners will be apply at the same time.
    The total length of this planner will be equal to the length of the sub-planners (which should all have the same
    length).
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        """
        Create a planner that will change the settings according to several sub-planners at the same time.
        Eg: At iteration the 5 with 2 sub-planners, the setting will be set as sub-planner[0][5] and sub-planner[1][5].
        It is not multi-thread computing.

        :param planners: The list of sub-planners to iterate in parallel, it can be any subclass of BasePlanner but they
        all need to have the same length.
        :param runs_basename: The basename of the run names generate by this planner. If defined, the names will be as:
        "basename-num" (eg: "plop-001", ""plop-002", "plop-003", ...). If this parameter is None or empty, the name will
        be generated according to the setting name, which is very descriptive but could be quite long.
        """
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for parallel planner')

        # Check planners length
        if not all(len(x) == len(planners[0]) for x in planners):
            raise ValueError('Impossible to run parallel planner if all sub-planners don\'t have the same length')

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # Iterate over every planners
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        super().__next__()
        sub_runs_name = [next(it) for it in self._planners_iterators]

        return self.format_name(sub_runs_name)

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return len(self.planners[0])

    def format_name(self, sub_runs_name: List[str]) -> str:
        """
        Format the name in case the basename is not set.
        As: "setting_name-value_setting_name-value" (eg: "epoch-15_learning_rate-0.01")

        :param sub_runs_name: The list of sub-planners names
        :return: The formatted name
        """
        # If no runs basename provided use the concatenation of all sub-planners names
        return self.basename() if self.runs_basename else '_'.join(sub_runs_name)

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()


class CombinatorPlanner(BasePlanner):
    """
    To organise the combination of planners.
    Each possible combination of values from planners will be used.
    The total length will be the product of the length of each sub-planners.
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        """
        Creat a combinator planner that will iterate over all possible combinations of sub-planners values.
        Eg: with sub-planner A iterates through [1, 2] abd sub-planner B iterates through [True, False], this planner
        will iterate through [A_1-B_True, A_2-B_True, A_1-B_False, A_2-B_False].

        :param planners: The list of sub-planners to iterate in parallel, it can be any subclass of BasePlanner but they
        all need to have the same length.
        :param runs_basename: The basename of the run names generate by this planner. If defined, the names will be as:
        "basename-num" (eg: "plop-001", ""plop-002", "plop-003", ...). If this parameter is None or empty, the name will
        be generated according to the setting name, which is very descriptive but could be quite long.
        """
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for combinator planner')

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)
        self._runs_name: List[Optional[str]] = [None] * len(planners)
        self._first_iter = True

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # Iterate over every planners
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        super().__next__()
        # For the first iteration, initialise every sub-planners with their first value
        if self._first_iter:
            self._first_iter = False
            self._runs_name = [next(it) for it in self._planners_iterators]
            return self.format_name()

        for i in range(len(self.planners)):
            try:
                self._runs_name[i] = next(self._planners_iterators[i])
                return self.format_name()
            except StopIteration:
                # If stop iteration trigger for the last sub-planner then the iteration is over and we let error
                # propagate.
                if i == (len(self.planners) - 1):
                    raise

                # If stop iteration trigger for an intermediate sub-planner, reset it and continue the loop
                self._planners_iterators[i] = iter(self.planners[i])
                self._runs_name[i] = next(self._planners_iterators[i])

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return math.prod(map(len, self.planners))

    def format_name(self) -> str:
        """
        Format the name in case the basename is not set.
        As: "setting_name-value_setting_name-value" (eg: "epoch-15_learning_rate-0.01")

        :return: The formatted name
        """
        # If no runs basename provided use the concatenation of all sub-planners names
        return self.basename() if self.runs_basename else '_'.join(self._runs_name)

    def reset_original_values(self):
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()
