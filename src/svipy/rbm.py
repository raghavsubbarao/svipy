import abc
from typing import Optional, Iterable, Union, overload, Tuple

import torch

from mlPlayGround.model import baseTorchModel, baseLossTracker


class restrictedBoltzmannMachine(baseTorchModel):
    def __init__(self, dims: Iterable[int], nHid: int, **kwargs) -> None:
        # initialization
        super(restrictedBoltzmannMachine, self).__init__(**kwargs)
        self.dims = dims

        self.W = torch.nn.Parameter(torch.zeros(*dims, nHid), requires_grad=True)
        self.vb = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.hb = torch.nn.Parameter(torch.zeros((nHid,)), requires_grad=True)

        k = torch.sqrt(6.0 / (torch.sum(torch.Tensor(dims)) + nHid))
        torch.nn.init.uniform_(self.W, -k, k)

        self.reconLossTracker = baseLossTracker(name="reconLoss")
        self.totalLossTracker = baseLossTracker(name="totalLoss")
        self.positiveLossTracker = baseLossTracker(name="positiveLoss")
        self.negativeLossTracker = baseLossTracker(name="negativeLoss")

    def pHidden(self, v):
        """
        Computes hidden state from the input.
        :param v: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_hidden)
        """
        return torch.sigmoid(torch.tensordot(v, self.W, dims=(tuple(range(1, len(self.dims) + 1)),
                                                              tuple(range(len(self.dims))))) + self.hb)

    def pVisible(self, h):
        """
        Computes visible state from hidden state.
        :param h: tensor of shape (batch_size, n_hidden)
        :return: tensor of shape (batch_size, n_visible)
        """
        return torch.sigmoid(torch.tensordot(h, self.W, dims=([1],[-1])) + self.vb)

    def reconstruct(self, x):
        """
        Computes visible state from the input. Reconstructs data.
        :param x: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_visible)
        """
        def sampleBernoulli(p):
            return torch.relu(torch.sign(p - torch.rand(p.shape)))

        return self.pVisible(sampleBernoulli(self.pHidden(x)))

    def freeEnergy(self, v):
        s = torch.tensordot(v, self.W, dims=(tuple(range(1, len(self.dims) + 1)),
                                             tuple(range(len(self.dims))))) + self.hb
        f1 = torch.sum(torch.log_(1. + torch.exp_(s)), dim=1)
        f2 = torch.sum(torch.flatten(v * self.vb, 1), 1)
        return - f1 - f2

    def energy(self, h, v):
        s = torch.tensordot(v, self.W, dims=(list(range(1, len(self.dims) + 1)),
                                             list(range(len(self.dims))))) + self.hb
        l1 = torch.sum(h * s, 1)
        l2 = torch.sum(torch.flatten(v * self.vb, 1), 1)
        return l1 + l2

    def gibbsSample(self, v, k=1):

        def sampleBernoulli(p):
            return torch.relu(torch.sign(p - torch.rand(p.shape)))

        h = self.pHidden(v)

        for i in range(k):
            v = self.pVisible(sampleBernoulli(h))
            h = self.pHidden(v)

        return h, v

    def forward(self, inputs):
        return self.pHidden(inputs)

    def trainStep(self, data, optimizer):
        X = data.to(self.device)
        _, v = self.gibbsSample(X, 1)

        positiveLoss = torch.mean(self.freeEnergy(data))
        negativeLoss = torch.mean(self.freeEnergy(v))
        totalLoss = positiveLoss - negativeLoss
        reconLoss = torch.mean(torch.sum(torch.flatten((data - v) ** 2, 1), 1))

        # Backpropagation
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        self.reconLossTracker.updateState(reconLoss)
        self.totalLossTracker.updateState(totalLoss)
        self.positiveLossTracker.updateState(positiveLoss)
        self.negativeLossTracker.updateState(negativeLoss)

        return {"reconLoss": self.reconLossTracker.result(),
                "totalLoss": self.totalLossTracker.result(),
                "positiveLoss": self.positiveLossTracker.result(),
                "negativeLoss": self.negativeLossTracker.result()}

    def validStep(self, data):
        X = data.to(self.device)
        _, v = self.gibbsSample(X, 1)

        positiveLoss = torch.mean(self.freeEnergy(data))
        negativeLoss = torch.mean(self.freeEnergy(v))
        totalLoss = positiveLoss - negativeLoss

        return totalLoss


if __name__ == "__main__":
    from torchvision import datasets
    from torchvision.transforms import ToTensor, Lambda
    from torch.utils.data import DataLoader

    class imageOnlyDataset(torch.utils.data.Dataset):
        '''
        Class to extract only the images as do do not need labels to train the VAE
        '''

        def __init__(self, original, index):
            self.__original = original
            self.__index = index

        @property
        def original(self):
            return self.__original

        @property
        def index(self):
            return self.__index

        def __len__(self):
            return len(self.original)

        def __getitem__(self, index):
            return self.original[index][self.index]

    thresh = Lambda(lambda x: torch.where(ToTensor()(x) > 0.5, 1.0, 0.0))
    onehot = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

    trainData = datasets.MNIST(root="../../../data", train=True, download=True, transform=thresh, target_transform=onehot)
    testData = datasets.MNIST(root="../../../data", train=False, download=True, transform=thresh, target_transform=onehot)

    imageData = imageOnlyDataset(trainData, 0)  # remove the labels as we dont need them
    print(len(trainData), len(testData), len(imageData))

    train_loader = DataLoader(trainData, batch_size=32, shuffle=False)  # shuffle=True)
    for x, _ in train_loader:
        break

    model = restrictedBoltzmannMachine((1, 28, 28), 64)
    model.pVisible(model.pHidden(x))