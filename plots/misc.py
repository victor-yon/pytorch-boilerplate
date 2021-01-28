from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.output import save_plot


def plot_losses(loss_evolution: List[float]) -> None:
    """
    Plot the evolution of the loss during the training.

    :param loss_evolution: A list of loss for each batch.
    """
    sns.relplot(data=loss_evolution, kind='line')
    # TODO x as nb train sample instead of batch number (and add batch size in title)
    # TODO add optional vertical lines for each epoch
    plt.title('Loss evolution')
    plt.xlabel('Batch number')
    plt.ylabel('Loss (Cross Entropy)')
    save_plot('loss')


def plot_confusion_matrix(nb_labels_predictions: np.ndarray, class_names: List[str] = None,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param class_names: The list of readable classes names
    :param annotations: If true the accuracy will be written in every cell
    """

    overall_accuracy = nb_labels_predictions.trace() / nb_labels_predictions.sum()
    rate_labels_predictions = nb_labels_predictions / nb_labels_predictions.sum(axis=1).reshape((-1, 1))

    # TODO dynamic font size
    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                annot=annotations,
                cbar=(not annotations))
    plt.title(f'Confusion matrix of {len(nb_labels_predictions)} classes '
              f'with {overall_accuracy * 100:.2f}% overall accuracy')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    save_plot('confusion_matrix')
