from typing import Tuple

from torch.nn import Module
from torch.utils.data import Dataset

from datasets.mock_classification_dataset import MockClassificationDataset
from networks.simple_classifier import SimpleClassifier
from utils.settings import settings


def load_datasets() -> Tuple[Dataset, Dataset]:
    """
    :return: The train and test datasets
    """
    # TODO Load your dataset here
    # Load the training dataset
    train_dataset = MockClassificationDataset(settings.nb_classes, settings.train_point_per_class)
    train_dataset.plot()  # Plot the data

    # Load the testing dataset
    test_dataset = MockClassificationDataset(settings.nb_classes, settings.test_point_per_class)

    return train_dataset, test_dataset


def build_network() -> Module:
    """
    :return: The neural network to use for the training and the testing
    """
    # TODO Create your neural network here
    return SimpleClassifier(input_size=2, nb_classes=settings.nb_classes)

# Execute "run.py" to start
