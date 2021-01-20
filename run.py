import random

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
from torch.utils.data import Dataset

from train import train
from utils.logger import logger
from utils.metrics import network_metrics
from utils.output import init_out_directory
from utils.settings import settings


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Setup console logger but wait to create the directory before to setup the file output
    logger.set_console_level(settings.logger_console_level)

    # Create the output directory to save results and plots
    init_out_directory()

    # Set plot style
    sns.set_theme()

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Print settings
    logger.info(settings)


def run(train_dataset: Dataset, test_dataset: Dataset, network: Module, device=None) -> None:
    """
    Run the training and the testing of the network.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param network: The neural network to train
    :param device: The device to use for pytorch (None = auto)
    """
    # Automatically chooses between CPU and GPU if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.debug(f'pyTorch device selected: {device}')

    # Send the network to the selected device (CPU or CUDA)
    network.to(device)
    # TODO send the dataset to device too

    # Save network stats and show if debug enable
    network_metrics(network, test_dataset[0][0].shape, device)

    # Start the training
    train(train_dataset, test_dataset, network)
