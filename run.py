import random

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from utils.logger import logger
from utils.metrics import network_metrics
from utils.output import init_out_directory
from utils.settings import settings


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Load logger
    logger.setLevel(settings.logger_output_level)

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Set plot style
    sns.set_theme()

    # Print settings
    logger.info(settings)

    # Create the output directory to save results and plots
    init_out_directory()


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

    # Send the network to the selected device (CPU or CUDA)
    network.to(device)
    # TODO send the dataset to device too

    # Save network stats and show if debug enable
    network_metrics(network, test_dataset[0][0].shape, device)

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=2)

    # Iterate epoch
    for epoch in range(settings.nb_epoch):
        # Iterate batches
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Run a training set for these data
            loss = network.training_step(inputs, labels)
