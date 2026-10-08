import abc
import copy

import torch
from tqdm.notebook import tqdm
from typing import Optional, List


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
    Linearly ramps a named attribute of the model from `start` to
    `end` over the first `warmupEpochs` epochs, then holds it at
    `end`. Originally used to combat posterior collapse in a VAE:
    but can be used for any named scalar attribute.
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
        Call once per epoch. Sets model.<param> to the current schedule value
        and returns it.
        """
        assert hasattr(model, self.param)
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

    def plot(self, axes: dict, **lineKwargs):
        """
        Draw each tracked metric's history (by index - epoch or batch,
        whichever this collection was updated at) onto a caller-supplied
        axes. Metrics with no matching key in `axes` are skipped, so several
        collections with only partially-overlapping metric names can share
        the same set of subplots.
        :param axes: dict mapping metric name -> matplotlib Axes to draw on.
        """
        for name, tracker in self.trackers.items():
            if name in axes:
                axes[name].plot(tracker.losses, **lineKwargs)

    @staticmethod
    def plotComparison(histories: dict, figsize=None, losses: Optional[List[str]] = None):
        """
        One subplot per metric name (union across all given collections), so
        metrics on very different scales (e.g. totalLoss vs klLoss) each get
        their own y-axis instead of collapsing onto a shared one.
        :param histories: named collections to overlay, e.g.
               {'train': model.epochTrackers['train'], 'valid': model.epochTrackers['valid']}
               figsize:
               losses:
        :return: (fig, axes) - axes is a dict keyed by metric name, so callers
                 can keep customizing individual subplots afterwards.
        """
        import matplotlib.pyplot as plt

        names = []
        for collection in histories.values():
            for name in collection.trackers:
                if name not in names and name in losses:
                    names.append(name)

        fig, axesList = plt.subplots(len(names), 1, figsize=figsize or (6, 3 * len(names)), squeeze=False)
        axes = {name: axesList[i, 0] for i, name in enumerate(names)}

        for splitName, collection in histories.items():
            collection.plot(axes, label=splitName)

        for name, ax in axes.items():
            ax.set_title(name)
            ax.set_xlabel('epoch')
            ax.legend()

        fig.tight_layout()
        return fig, axes


class baseTorchModel(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(baseTorchModel, self).__init__(*args, **kwargs)

        # todo: do we need per-batch tracking? remove if not required
        self.trainTrackers = lossTrackerCollection()  # per batch
        self.epochTrackers = {'train': lossTrackerCollection(),
                              'valid': lossTrackerCollection()}  # per epoch

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

    def trainStep(self, data, optimizer, gradientClippingNorm=None):
        losses = self.computeLoss(data)

        optimizer.zero_grad()
        losses['totalLoss'].backward()
        if gradientClippingNorm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), gradientClippingNorm)
        optimizer.step()

        return self.trainTrackers.update({name: loss.detach() for name, loss in losses.items()})

    @torch.no_grad()
    def validStep(self, data):
        return self.computeLoss(data)

    def trainLoop(self,
                  trainDataLoader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  epochs: int, reportIters: int = 100, verbose=False,
                  validDataLoader: Optional[torch.utils.data.DataLoader] = None,
                  checkpointPath: Optional[str] = None, checkPointName: Optional[str] = None,
                  scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                  earlyStopper: Optional[earlyStopping] = None,
                  annealers: Optional[paramAnnealer] = None,
                  gradientClippingNorm: Optional[float] = None):

        if earlyStopper is not None and validDataLoader is None:
            raise ValueError("earlyStopping requires a validDataLoader to monitor")

        annealers = annealers or []
        trainSize = len(trainDataLoader.dataset)

        self.trainTrackers.clear()
        self.epochTrackers['train'].clear()
        self.epochTrackers['valid'].clear()

        pbar = range(epochs) if verbose else tqdm(range(epochs))
        for t in pbar:
            if verbose:
                print(f"Epoch {t + 1}\n-------------------------------")
            else:
                # pbar.set_postfix({'epoch': t+1})
                pass

            for annealer in annealers:
                current = annealer.step(t, self)
                if verbose:
                    print(f"{annealer.param}: {current:>5f}")

            # Set the model to training mode - do here
            # in case theres a validation dataset
            self.train()

            trainTotals, nTrainBatches = {}, 0
            pbarDataLoader = trainDataLoader if verbose else tqdm(trainDataLoader, leave=False)
            for batch, data in enumerate(pbarDataLoader):
                metrics = self.trainStep(data, optimizer, gradientClippingNorm)
                for name, value in metrics.items():
                    trainTotals[name] = trainTotals.get(name, 0.0) + value.item()
                nTrainBatches += 1

                if verbose:
                    if (batch + 1) % reportIters == 0:
                        print(' '.join([f'{metric}: {metrics[metric]:0.5f}' for metric in metrics]) +
                              f'[{(batch + 1) * trainDataLoader.batch_size:>5d}/{trainSize:>5d}]')
                else:
                    if (batch + 1) % reportIters == 0:
                        pbarDataLoader.set_postfix({metric: f'{metrics[metric]:>0.3f}' for metric in metrics})

            self.epochTrackers['train'].update({name: total / nTrainBatches for name, total in trainTotals.items()})

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
                self.epochTrackers['valid'].update(validMetrics)

                if verbose:
                    print(f"Validation Error: {validationLoss:>7f}")
                else:
                    pbar.set_postfix({"validation loss": f'{validationLoss:>7f}'})

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

class reshape(torch.nn.Module):
    def __init__(self, shape):
        super(reshape, self).__init__()
        self.__shape = shape

    def forward(self, inputs: torch.tensor):
        return inputs.view(-1, *self.__shape)
