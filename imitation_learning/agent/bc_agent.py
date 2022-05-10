import numpy as np
import torch
from torch.optim import Adam, SGD , Adamax
from agent.networks import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BCAgent:

    def __init__(self, lr=1e-4):
        # TODO: Define network, loss function, optimizer
        self.net = CNN().to(device)
        self.optimizer = Adamax(self.net.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = torch.tensor(X_batch, dtype=torch.float32).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)
        # TODO: forward + backward + optimize
        self.optimizer.zero_grad()
        outputs = self.net.forward(X_batch)
        # outputs.to(device)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()

        return outputs, y_batch, loss

    def predict(self, X):
        # TODO: forward pass
        # with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).detach().to(device)
        outputs = self.net.forward(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

