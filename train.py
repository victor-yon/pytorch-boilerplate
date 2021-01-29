from typing import List

import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_losses
from utils.logger import logger
from utils.output import load_network, save_network, save_results
from utils.settings import settings
from utils.timer import SectionTimer


def train(train_dataset: Dataset, test_dataset: Dataset, network: Module) -> None:
    # If path set, try to load a pre trained network from cache
    if settings.trained_network_cache_path and load_network(network, settings.trained_network_cache_path):
        return  # Stop here if the network parameters are successfully loaded from cache file

    # Turn on the training mode of the network
    network.train()

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=2)
    nb_batch = len(train_loader)

    # Store the loss values for plot
    loss_evolution: List[float] = []
    epochs_stats: List[dict] = []

    with SectionTimer('network training'):
        # Iterate epoch
        for epoch in range(settings.nb_epoch):
            # Iterate batches
            for i, data in enumerate(train_loader):
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # Run a training set for these data
                loss = network.training_step(inputs, labels)
                loss_evolution.append(float(loss))
                # TODO Log progress and loss based on time interval in debug
                # TODO Print a visual loading bar, disable with settings

            # TODO Test between each epoch, disable with a setting
            # Epoch statistics
            _record_epoch_stats(epochs_stats, loss_evolution[-len(train_loader):])

    save_results(epochs_stats=epochs_stats)

    # Post train plots
    plot_losses(loss_evolution)

    if settings.save_network:
        save_network(network, 'trained_network')


def _record_epoch_stats(epochs_stats: List[dict], epoch_losses: List[float]) -> None:
    """
    Record the statics for one epoch.

    :param epochs_stats: The list where to store the stats, append in place.
    :param epoch_losses: The losses list of the current epoch.
    """
    stats = dict()
    stats['losses_mean'] = float(np.mean(epoch_losses))

    # Compute the loss difference with the previous epoch
    losses_mean_diff = 0
    if len(epochs_stats) > 0:
        losses_mean_diff = stats['losses_mean'] - epochs_stats[-1]['losses_mean']
    stats['losses_mean_diff'] = losses_mean_diff

    stats['losses_std'] = float(np.std(epoch_losses))

    epochs_stats.append(stats)

    epoch_num = len(epochs_stats)
    logger.info(f"Epoch {epoch_num:3}/{settings.nb_epoch} ({epoch_num / settings.nb_epoch:7.2%}) "
                f"| loss: {stats['losses_mean']:.5f} "
                f"| diff: {stats['losses_mean_diff']:+.5f} "
                f"| std: {stats['losses_std']:.5f}")
