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
                 'totalLoss' key - the value backpropagated during
                 training and monitored for checkpointing / early stopping.
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
                  earlyStopper=None,
                  annealers=None):

        if earlyStopper is not None and validDataLoader is None:
            raise ValueError("earlyStopping requires a validDataLoader to monitor")

        annealers = annealers or []
        trainSize = len(trainDataLoader.dataset)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            for annealer in annealers:
                current = annealer.step(t, self)
                print(f"{annealer.param}: {current:>7f}")

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

            stillAnnealing = any(not annealer.isDone(t) for annealer in annealers)
            if earlyStopper is not None and not stillAnnealing:
                if earlyStopper.step(validationLoss, t, self):
                    print(f"Early stopping: no improvement in {earlyStopper.patience} epochs "
                          f"(best={earlyStopper.best:>7f} @ epoch {earlyStopper.bestEpoch + 1})")
                    break

        if earlyStopper is not None:
            earlyStopper.restore(self)


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


class paramAnnealer:
    """
    Linearly ramps a named attribute on a model from `start` up to `end`
    over the first `warmupEpochs` epochs, then holds it at `end`. Originally
    written for annealing a VAE's `beta` (KL weight) to combat posterior
    collapse - giving the decoder a head start relying on the latent code
    before the KL term reaches full strength - but works for any named
    scalar attribute on any baseTorchModel.
    """
    def __init__(self, param: str, start: float, end: float, warmupEpochs: int):
        assert warmupEpochs > 0
        self.param = param
        self.start = start
        self.end = end
        self.warmupEpochs = warmupEpochs

    def value(self, epoch: int) -> float:
        if epoch >= self.warmupEpochs - 1:
            return self.end
        return self.start + (self.end - self.start) * (epoch / (self.warmupEpochs - 1))

    def isDone(self, epoch: int) -> bool:
        return epoch >= self.warmupEpochs - 1

    def step(self, epoch: int, model: torch.nn.Module) -> float:
        """
        Call once per epoch. Sets model.<param> to the current schedule
        value and returns it.
        """
        current = self.value(epoch)
        setattr(model, self.param, current)
        return current


class baseLossTracker:
    def __init__(self, name):
        self.__name = name
        self.__losses = []

    @property
    def losses(self):
        return self.__losses

    def clear(self):
        self.__losses = []

    def updateState(self, loss):
        self.__losses.append(loss)

    def result(self):
        return self.__losses[-1]


class lossTrackerCollection:
    """
    Lazily creates and updates a baseLossTracker per named loss, so models
    don't need to hand-declare one tracker per loss term.
    """
    def __init__(self):
        self.__trackers = {}

    @property
    def trackers(self):
        return self.__trackers

    @property
    def metrics(self):
        return list(self.__trackers.values())

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


class reshape(torch.nn.Module):
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.__shape = shape

    def forward(self, inputs: torch.tensor):
        return inputs.view(-1, *self.__shape)
