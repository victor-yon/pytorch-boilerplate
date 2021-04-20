import gc
import random

import numpy as np
import torch
from codetiming import Timer

from main import build_network, load_datasets
from test import test
from train import train
from utils.logger import logger
from utils.metrics import network_metrics
from utils.output import init_out_directory, save_results, save_timers, set_plot_style
from utils.settings import settings
from utils.timer import SectionTimer


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Setup console logger but wait to create the directory before to setup the file output
    logger.set_console_level(settings.logger_console_level)

    # Create the output directory to save results and plots
    init_out_directory()

    if settings.is_named_run():
        logger.info(f'Run name: {settings.run_name}')

    # Set plot style
    set_plot_style()

    if settings.seed is not None:
        # Set random seeds for reproducibility
        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)

    # Print settings
    logger.debug(settings)


def clean_up() -> None:
    """
    Clean up the environment after all operations. After that a new run can start again.
    """

    # Save recorded timers in a file
    save_timers()
    # Clear all timers
    Timer.timers.clear()

    # Disable the log file, so a new one can be set later
    if settings.is_named_run() and settings.logger_file_enable:
        logger.disable_log_file()

    # Free CUDA memory
    gc.collect()
    torch.cuda.empty_cache()


def start_run() -> None:
    """
    Run the training and the testing of the network.
    Run steps:
        - preparation
        - load datasets
        - build network
        - send data to device (GPU/CPU)
        - train network
        - test network
        - clean up
    """

    # Prepare the environment
    preparation()

    # Catch and log every exception during the runtime
    # noinspection PyBroadException
    try:
        with SectionTimer('run'):
            with SectionTimer('datasets loading', 'debug'):
                # Load dataset from user function
                train_dataset, test_dataset = load_datasets()

            # Build network from user function
            network = build_network()

            # Automatically chooses between CPU and GPU if not specified
            if settings.device is None or settings.device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(settings.device)

            logger.debug(f'pyTorch device selected: {device}')

            # Send the network and the datasets to the selected device (CPU or CUDA)
            # We assume the GPU have enough memory to store the whole network and datasets. If not it should be split.
            network.to(device)
            train_dataset.to(device)
            test_dataset.to(device)

            # Save network stats and show if debug enable
            network_metrics(network, test_dataset[0][0].shape, device)

            # Start the training
            train(network, train_dataset, test_dataset, device)

            # Start normal test
            test(network, test_dataset, device, final=True)

            # Arrived to the end successfully (no error)
            save_results(success_run=True)

    except KeyboardInterrupt:
        logger.error('Run interrupted by the user.')
        raise  # Let it go to stop the runs planner if needed
    except Exception:
        logger.critical('Run interrupted by an unexpected error.', exc_info=True)
        # TODO deal with this error in runs planner (eg. stop after a count)
    finally:
        # Clean up the environment, ready for a new run
        del train_dataset, test_dataset, network
        clean_up()


if __name__ == '__main__':
    start_run()
