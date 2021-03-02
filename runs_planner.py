from main import main
from utils.logger import logger
from utils.planner import BasePlanner, CombinatorPlanner, ParallelPlanner, Planner
from utils.settings import settings


def start_planner(runs_planner: BasePlanner):
    # TODO add train once option
    # TODO skip existing runs names
    # TODO add loading bar
    # TODO do a checking (names and values) before to start to avoid error during the run

    # Typical settings override for planner runs
    settings.run_name = None  # The name should be override during the runs
    settings.visual_progress_bar = False
    settings.show_images = False

    logger.info(f'Starting a set of {len(runs_planner)} runs with a planner')

    # At every iteration of the loop the settings will be update according to the planner current state
    for run_name in runs_planner:
        # Set the name of this run according to the planner
        # All other settings are already set during the "next" operation
        settings.run_name = run_name
        # Start the run
        main()


if __name__ == '__main__':
    # Create the planner by defining the settings configurations evolution during the runs
    planner = CombinatorPlanner([
        ParallelPlanner([
            Planner('train_point_per_class', range(500, 701, 100), runs_basename='nb_train'),
            Planner('test_point_per_class', range(200, 401, 100), runs_basename='nb_test'),
        ]),
        Planner('nb_epoch', [1, 5, 10])
    ])

    start_planner(planner)
