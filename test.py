import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_confusion_matrix
from utils.logger import logger
from utils.output import save_results
from utils.settings import settings


def test(test_dataset: Dataset, network: Module) -> float:
    """
    Start testing inference on a dataset.

    :param test_dataset: The testing dataset
    :param network: The network to use
    :return: The overall accuracy
    """
    logger.info('Start network testing...')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    nb_batch = len(test_loader)
    nb_classes = len(test_dataset.classes)

    nb_correct = 0
    nb_total = 0
    nb_labels_predictions = np.zeros((nb_classes, nb_classes))
    # Diable gradient for performances
    with torch.no_grad():
        # Iterate batches
        for i, data in enumerate(test_loader):
            # Get the inputs: data is a list of [inputs, labels]
            inputs, labels = data

            # Forward
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max value for each image of the batch

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()
            for label, pred in zip(labels, predicted):
                nb_labels_predictions[label][pred] += 1

    accuracy = float(nb_correct / nb_total)
    classes_accuracy = [float(l[i] / np.sum(l)) for i, l in enumerate(nb_labels_predictions)]
    logger.info(f'Test overall accuracy: {accuracy * 100:05.2f}%')
    logger.info(f'Test accuracy per classes:\n\t' +
                "\n\t".join([f'{test_dataset.classes[i]}: {a * 100:05.2f}%' for i, a in enumerate(classes_accuracy)]))

    logger.info('Network testing competed')

    results = {f'accuracy': accuracy, f'classes_accuracy': classes_accuracy}
    save_results(**results)
    plot_confusion_matrix(nb_labels_predictions, class_names=test_dataset.classes)

    return accuracy
