import matplotlib.pyplot as plt
import seaborn as sns

from utils.output import save_plot


def plot_dataset(features, labels, nb_classes: int) -> None:
    """
    Create a plot that represent the dataset and show it.

    :param features: The list of data features (as a 2D torch array)
    :param labels: The list of data labels (as a 1D torch array)
    :param nb_classes: The total number of classes
    """
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, markers=True)
    plt.title(f'Data from {nb_classes} classes')
    save_plot('mock_dataset')
