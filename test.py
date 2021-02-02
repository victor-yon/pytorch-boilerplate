import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_confusion_matrix
from utils.logger import logger
from utils.output import save_results
from utils.settings import settings
from utils.timer import SectionTimer


def test(network: Module, test_dataset: Dataset, test_name: str = '', final: bool = False, limit: int = 0) -> float:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
    :return: The overall accuracy.
    """

    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    nb_classes = len(test_dataset.classes)

    nb_correct = 0
    nb_total = 0
    nb_labels_predictions = np.zeros((nb_classes, nb_classes))

    # Diable gradient for performances
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'):
        # Iterate batches
        for i, (inputs, labels) in enumerate(test_loader):
            # Stop testing after the limit
            if limit and i * settings.batch_size >= limit:
                break

            # Forward
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max value for each image of the batch

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()
            for label, pred in zip(labels, predicted):
                nb_labels_predictions[label][pred] += 1

    accuracy = float(nb_correct / nb_total)

    # Give more information for the final test
    if final:
        classes_accuracy = [float(l[i] / np.sum(l)) for i, l in enumerate(nb_labels_predictions)]
        logger.info(f'Test overall accuracy: {accuracy:05.2%}')
        logger.info(f'Test accuracy per classes:\n\t' +
                    "\n\t".join(
                        [f'{test_dataset.classes[i]}: {a:05.2%}' for i, a in enumerate(classes_accuracy)]))

        save_results(final_accuracy=accuracy, final_classes_accuracy=classes_accuracy)
        plot_confusion_matrix(nb_labels_predictions, class_names=test_dataset.classes)

    return accuracy
