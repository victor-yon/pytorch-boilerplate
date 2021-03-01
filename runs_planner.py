from main import main
from utils.planner import CombinatorPlanner, Planner

if __name__ == '__main__':
    # TODO add train once option
    # TODO skip existing runs names

    # planner = Planner('train_point_per_class', range(50, 501, 50), 'train-size')
    planner = CombinatorPlanner([
        Planner('train_point_per_class', range(250, 501, 50)),
        Planner('nb_epoch', range(1, 4, 1)),
    ])

    print(len(planner))

    for run_name in planner:
        # At every iteration of the loop the settings will be update according to the planner current state
        # Start the run
        main()
