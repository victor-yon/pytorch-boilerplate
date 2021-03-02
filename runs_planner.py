from main import main
from utils.planner import Planner, SequencePlanner
from utils.settings import settings

if __name__ == '__main__':
    # TODO add train once option
    # TODO skip existing runs names
    # TODO add a default settings file for run planner
    # TODO add loading bar
    # TODO do a checking (names and values) before to start to avoid error during the run

    # planner = Planner('train_point_per_class', range(50, 501, 50))
    planner = SequencePlanner([
        Planner('train_point_per_class', range(250, 501, 50)),
        Planner('nb_epoch', range(1, 3, 1), runs_basename='e'),
    ])

    print(len(planner))

    # At every iteration of the loop the settings will be update according to the planner current state
    for run_name in planner:
        # Set the name of this run according to the planner
        # All other settings are already set during the "next" operation
        settings.run_name = run_name
        # Start the run
        main()
