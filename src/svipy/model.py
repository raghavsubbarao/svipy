import abc
import torch
# import numpy as np

class baseTorchModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(baseTorchModel, self).__init__(*args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    def fit(self, dataset, optimizer, epochs):
        for t in range(epochs):
            train_loop(dataset, optimizer)
            test_loop(test_dl,)

    def trainLoop(self, trainDataLoader, optimizer, epochs,
                  reportIters=100,
                  scheduler=None,
                  checkpointPath=None, checkPointName=None,
                  validDataLoader=None):

        trainSize = len(trainDataLoader.dataset)
        bestValidation = None

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            # Set the model to training mode - do here
            # in case theres a validation dataset
            self.train()

            for batch, data in enumerate(trainDataLoader):
                metrics = self.trainStep(data, optimizer)

                if (batch + 1) % reportIters == 0:
                    print(' '.join([f'{l}: {metrics[l]:>7f}' for l in metrics]) +
                          f'[{(batch + 1) * trainDataLoader.batch_size:>5d}/{trainSize:>5d}]')

            if validDataLoader:
                self.eval()

                validationLoss = 0
                for data in validDataLoader:
                    validationLoss += self.validStep(data)

                validationLoss /= len(validDataLoader)

                if bestValidation is None or bestValidation > validationLoss:
                    bestValidation = validationLoss

                print(f"Validation Error: {validationLoss:>7f}")

            if checkpointPath:
                assert(checkPointName is not None)
                modelDict = {'epoch': t, 'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                if validDataLoader:
                    modelDict['validation_loss'] = validationLoss
                torch.save(modelDict, checkpointPath + f'{checkPointName}-{t}.model')

            if scheduler:
                scheduler.step()

    @abc.abstractmethod
    def trainStep(self, data, optimizer):
        pass

    @abc.abstractmethod
    def validStep(self, data):
        pass


class baseLossTracker:
    def __init__(self, name):
        self.__name = name
        self.__losses = []

    def clear(self):
        self.__losses = []

    def updateState(self, loss):
        self.__losses.append(loss)

    def result(self):
        return self.__losses[-1]


class reshape(torch.nn.Module):
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.__shape = shape

    def forward(self, inputs: torch.tensor):
        return inputs.view(-1, *self.__shape)
