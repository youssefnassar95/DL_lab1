import torch.nn as nn
import torch
import torch.nn.functional as F




"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=4):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network

        self.cnn_layers = nn.Sequential(
            torch.nn.Conv2d(history_length, n_classes*8, kernel_size=7),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(n_classes * 8, n_classes * 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(n_classes * 16, n_classes * 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(n_classes*32, n_classes * 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.linear_layers = nn.Sequential(
            # input from sequential conv layers
            torch.nn.Linear(32 * n_classes * 5 * 5, n_classes),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # TODO: compute forward pass
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x,0).unsqueeze(0)
        x = self.linear_layers(x)

        return x

