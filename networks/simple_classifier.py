from typing import Any

import torch.nn as nn
import torch.nn.functional as f
from torch import optim


class SimpleClassifier(nn.Module):
    def __init__(self, input_size: int, nb_classes: int):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)  # Input -> Hidden 1
        self.fc2 = nn.Linear(50, 50)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(50, nb_classes)  # Hidden 2 -> Output

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, inputs: Any, labels: Any):
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()

        return loss
