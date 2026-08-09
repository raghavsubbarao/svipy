import abc
import copy

import torch


class baseTorchModel(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(baseTorchModel, self).__init__(*args, **kwargs)
        self.trainTrackers = lossTrackerCollection()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def metrics(self):
        return self.trainTrackers.metrics

    @abc.abstractmethod
    def computeLoss(self, data) -> dict:
        """
        Compute the losses for a single batch of data.
        :param data: a batch as produced by the DataLoader
        :return: dict of named scalar-tensor losses. Must include a
        'totalLoss' key - the value backpropagated during training and
        monitored for checkpointing / early stopping.
        """
        pass

    def trainStep(self, data, optimizer):
        losses = self.computeLoss(data)

        optimizer.zero_grad()
        losses['totalLoss'].backward()
        optimizer.step()

        return self.trainTrackers.update({name: loss.detach() for name, loss in losses.items()})

    @torch.no_grad()
    def validStep(self, data):
        return self.computeLoss(data)

    def trainLoop(self, trainDataLoader, optimizer, epochs,
                  reportIters=100,
                  scheduler=None,
                  checkpointPath=None, checkPointName=None,
                  validDataLoader=None,
                  earlyStopping=None):

        if earlyStopping is not None and validDataLoader is None:
            raise ValueError("earlyStopping requires a validDataLoader to monitor")

        trainSize = len(trainDataLoader.dataset)

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

            validationLoss = None
            if validDataLoader:
                self.eval()

                totals, nBatches = {}, 0
                for data in validDataLoader:
                    losses = self.validStep(data)
                    for name, loss in losses.items():
                        totals[name] = totals.get(name, 0.0) + loss.item()
                    nBatches += 1

                validMetrics = {name: total / nBatches for name, total in totals.items()}
                validationLoss = validMetrics['totalLoss']

                print(f"Validation Error: {validationLoss:>7f}")

            if checkpointPath:
                assert(checkPointName is not None)
                modelDict = {'epoch': t, 'model_state_dict': self.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict()}
                if validationLoss is not None:
                    modelDict['validation_loss'] = validationLoss
                torch.save(modelDict, checkpointPath + f'{checkPointName}-{t}.model')

            if scheduler:
                scheduler.step()

            if earlyStopping is not None:
                if earlyStopping.step(validationLoss, t, self):
                    print(f"Early stopping: no improvement in {earlyStopping.patience} epochs "
                          f"(best={earlyStopping.best:>7f} @ epoch {earlyStopping.bestEpoch + 1})")
                    break

        if earlyStopping is not None:
            earlyStopping.restore(self)


class earlyStopping:
    """
    Monitors the 'totalLoss' returned by validStep across epochs and signals
    trainLoop to stop once it fails to improve by at least minDelta for
    `patience` consecutive epochs. Optionally snapshots the best weights seen
    and restores them once training stops.
    """
    def __init__(self, patience: int = 10, minDelta: float = 0.0,
                 mode: str = 'min', restoreBestWeights: bool = True):
        assert mode in ('min', 'max')
        self.patience = patience
        self.minDelta = minDelta
        self.mode = mode
        self.restoreBestWeights = restoreBestWeights

        self.best = None
        self.bestEpoch = None
        self.numBadEpochs = 0
        self.__bestState = None

    def __isImprovement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == 'min':
            return current < self.best - self.minDelta
        return current > self.best + self.minDelta

    def step(self, current: float, epoch: int, model: torch.nn.Module) -> bool:
        """
        Call once per epoch with the monitored validation loss.
        :return: True if training should stop.
        """
        if self.__isImprovement(current):
            self.best = current
            self.bestEpoch = epoch
            self.numBadEpochs = 0
            if self.restoreBestWeights:
                self.__bestState = copy.deepcopy(model.state_dict())
        else:
            self.numBadEpochs += 1

        return self.numBadEpochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.restoreBestWeights and self.__bestState is not None:
            model.load_state_dict(self.__bestState)


class baseLossTracker:
    def __init__(self, name):
        self.__name = name
        self.__value = None

    def clear(self):
        self.__value = None

    def updateState(self, loss):
        self.__value = loss

    def result(self):
        return self.__value


class lossTrackerCollection:
    """
    Lazily creates and updates a baseLossTracker per named loss, so models
    don't need to hand-declare one tracker per loss term.
    """
    def __init__(self):
        self.__trackers = {}

    def update(self, losses: dict) -> dict:
        result = {}
        for name, value in losses.items():
            if name not in self.__trackers:
                self.__trackers[name] = baseLossTracker(name)
            self.__trackers[name].updateState(value)
            result[name] = self.__trackers[name].result()
        return result

    def clear(self):
        for tracker in self.__trackers.values():
            tracker.clear()

    @property
    def metrics(self):
        return list(self.__trackers.values())


class reshape(torch.nn.Module):
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.__shape = shape

    def forward(self, inputs: torch.tensor):
        return inputs.view(-1, *self.__shape)
