import math
from typing import Any, Collection, Iterator, List, Optional

from utils.settings import settings


class BasePlanner:
    """
    Abstract class Base Planner.
    Use one of its children class.
    """

    def __init__(self, runs_basename: str = ''):
        self.runs_basename = runs_basename
        self.num_count = 0

    def basename(self):
        return f'{self.runs_basename}-{self.num_count:03d}'

    def __iter__(self) -> Iterator:
        raise NotImplemented('This abstract class need to override iteration an length methods')

    def __next__(self):
        self.num_count += 1

    def __len__(self) -> int:
        raise NotImplemented('This abstract class need to override iteration an length methods')


class Planner(BasePlanner):
    """
    Simple planner use to start a set of runs with a different values for one setting.
    """

    def __init__(self, setting_name: str, setting_values: Collection, runs_basename: str = ''):
        super().__init__(runs_basename)

        self.setting_name = setting_name
        self.setting_values = setting_values
        self._values_iterator = None

    def __iter__(self) -> Iterator:
        self._values_iterator = iter(self.setting_values)
        return self

    def __next__(self) -> str:
        super().__next__()
        # Get new value
        value = next(self._values_iterator)

        # Set new value
        setattr(settings, self.setting_name, value)

        # Return the name of this run
        return self.format_name(value)

    def __len__(self) -> int:
        return len(self.setting_values)

    def format_name(self, value: Any) -> str:
        # If no runs basename provided use the variable name and value as run name
        return self.basename() if self.runs_basename else f'{self.setting_name}-{value}'


class SequencePlanner(BasePlanner):
    """
    To organise planners by sequential order.
    When the current planner is over the next one on the list will start.
    The total length of this sequence will be the sum of each sub-planners.
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for sequence planner')

        self.planners = planners
        self._planners_iterator = None
        self._current_planner_iterator = None

    def __iter__(self):
        # First iterate over every planners
        self._planners_iterator = iter(self.planners)
        # Then iterate inside each planner
        self._current_planner_iterator = iter(next(self._planners_iterator))

        return self

    def __next__(self):
        super().__next__()
        try:
            # Try to iterate inside the current planner
            sub_run_name = next(self._current_planner_iterator)
            return self.format_name(sub_run_name)
        except StopIteration:
            # FIXME reset settings here
            # If current planner is over, open the next one
            # If it's already the last planner then the StopIteration will be raise again here
            self._current_planner_iterator = iter(next(self._planners_iterator))

            # Recursive call with the new planner
            return next(self)

    def __len__(self):
        return sum(map(len, self.planners))

    def format_name(self, sub_run_name: str) -> str:
        # If no runs basename provided use the name of the last sub-planner
        return self.basename() if self.runs_basename else sub_run_name


class ParallelPlanner(BasePlanner):
    """
    To organise planners in parallel.
    All planners will be apply at the same time.
    The total length of this planner will be equal to the length of the sub-planners (which should all have the same
    length).
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for parallel planner')

        # Check planners length
        if not all(len(x) == len(planners[0]) for x in planners):
            raise ValueError('Impossible to run parallel planner if all sub-planners don\'t have the same length')

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)

    def __iter__(self):
        # Iterate over every planners
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
        super().__next__()
        sub_runs_name = [next(it) for it in self._planners_iterators]

        return self.format_name(sub_runs_name)

    def __len__(self):
        return len(self.planners[0])

    def format_name(self, sub_runs_name: List[str]) -> str:
        # If no runs basename provided use the concatenation of all sub-planners names
        # As: "planner-1_planner-2_planner-3"
        return self.basename() if self.runs_basename else '_'.join(sub_runs_name)


class CombinatorPlanner(BasePlanner):
    """
    To organise the combination of planners.
    Each possible combination of values from planners will be used.
    The total length will be the product of the length of each sub-planners.
    """

    def __init__(self, planners: List[BasePlanner], runs_basename: str = ''):
        super().__init__(runs_basename)

        if len(planners) == 0:
            raise ValueError('Empty planners list for combinator planner')

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)
        self._runs_name: List[Optional[str]] = [None] * len(planners)
        self._first_iter = True

    def __iter__(self):
        # Iterate over every planners
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
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
        return math.prod(map(len, self.planners))

    def format_name(self) -> str:
        # If no runs basename provided use the concatenation of all sub-planners names
        # As: "planner-1_planner-2_planner-3"
        return self.basename() if self.runs_basename else '_'.join(self._runs_name)
