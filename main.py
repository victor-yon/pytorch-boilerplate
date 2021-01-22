from datasets.mock_classification_dataset import MockClassificationDataset
from networks.simple_classifier import SimpleClassifier
from run import preparation, run, clean_up
from utils.settings import settings


def main():
    # Prepare the environment
    preparation()

    # Load the training dataset
    train_set = MockClassificationDataset(settings.nb_classes, settings.train_point_per_class)
    train_set.show_plot()  # Plot and show the data

    # Load test testing dataset
    test_set = MockClassificationDataset(settings.nb_classes, settings.test_point_per_class)

    # Build the network
    net = SimpleClassifier(input_size=2, nb_classes=len(train_set.classes))

    # Run the training and the test
    run(train_set, test_set, net)

    # Clean up the environment, ready for a new run
    clean_up()


if __name__ == '__main__':
    main()
