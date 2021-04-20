from typing import Any

import torch.nn as nn
import torch.nn.functional as f
from torch import optim

from utils.settings import settings


class SimpleClassifier(nn.Module):
    """
    Simple classifier neural network.
    Should be use as an example.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        layers_size = [input_size]
        layers_size.extend(settings.hidden_layers_size)
        layers_size.append(nb_classes)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layers_size) - 1):
            self.fc_layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))

        # Convert the tensor to long before to call the CrossEntropy to match with the expected data type.
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        # Use relu function on every layer except the last one
        for fc in self.fc_layers[:-1]:
            x = f.relu(fc(x))

        return self.fc_layers[-1](x)

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels.long())
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def get_loss_name(self) -> str:
        """
        :return: The name of the loss function (criterion).
        """
        return type(self._criterion).__name__

    def get_optimizer_name(self) -> str:
        """
        :return: The name of the optimiser function.
        """
        return type(self._optimizer).__name__
