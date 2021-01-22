from main import main
from utils.logger import logger
from utils.settings import settings

if __name__ == '__main__':
    planning = {
        'train_point_per_class': {'name': 'train-size', 'range': range(10, 501, 10)}
    }

    # Iterate through all planning settings and values
    for setting, plan in planning.items():
        for i, value in enumerate(plan['range'], start=1):
            logger.info(f'Start {setting} with value {value}')
            # Change the settings
            settings.run_name = f"{plan['name']}-{i:03}"
            setattr(settings, setting, value)
            # Start the run
            main()
