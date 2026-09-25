import abc
import math
import numpy as np

from typing import Optional, Iterable, Union, overload, Tuple, List

import torch
from torchdiffeq import odeint_adjoint

from svipy.model import baseTorchModel


#################################
#           Norm Flow           #
#################################
class normFlowModule(torch.nn.Module, abc.ABC):
    """
    Base class for normalizing flow modules. Subclasses must implement
    forwardLogDetJacobian and forward (z→x direction).
    """

    @abc.abstractmethod
    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        :param y: B x ... tensor where B=batch_size
        :return: tensor of size (B,) of log determinant of the Jacobian dx/dz
        """
        pass


class normFlowSequential(torch.nn.Sequential, normFlowModule):

    @overload
    def __init__(self, *args: normFlowModule) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, normFlowModule]") -> None:
        ...

    def __init__(self, *args) -> None:
        super(normFlowSequential, self).__init__(*args)

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        ll = torch.zeros(y.shape[0])
        for module in self:
            ll = ll + module.forwardLogDetJacobian(y)
            y, _ = module(y)
        return ll

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ll = []
        for module in self:
            y, fldj = module(y)
            ll.append(fldj)
        return y, torch.sum(torch.stack(ll, -1), 1)


class normFlowPrior(abc.ABC):
    def __init__(self, flow: normFlowModule, dim: int):
        self.flow = flow
        self.dim = dim

    @abc.abstractmethod
    def baseLikelihood(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def logLikelihood(self, z: torch.Tensor) -> torch.Tensor:
        u, logDet = self.flow(z)
        logBase = self.baseLikelihood(u)
        return logBase + logDet

class normFlowPriorUniform(normFlowPrior):
    def __init__(self, flow: normFlowModule, dim: int):
        super(normFlowPriorUniform, self).__init__(flow, dim)

    def baseLikelihood(self, u):
        return 0

class normFlowPriorNormal(normFlowPrior):
    def __init__(self, flow: normFlowModule, dim: int):
        super(normFlowPriorNormal, self).__init__(flow, dim)

    def baseLikelihood(self, u):
        return -0.5 * (u.pow(2).flatten(start_dim=1).sum(dim=1) + self.dim * math.log(2 * math.pi))


class normFlowPosterior:
    def __init__(self, flow: normFlowModule, dim: int):
        self.flow = flow
        self.dim = dim

    def logDetJacobian(self, u: torch.Tensor) -> torch.Tensor:
        _, logDet = self.flow(u)
        return -logDet

#################################
#            Real NVP           #
#################################
class nvpBatchNorm2d(torch.nn.modules.BatchNorm2d, normFlowModule):
    """
    Batch norm for real NVP coupling layers. Tracks current batch variance
    during training for correct log-det computation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True, device=None, dtype=None) -> None:
        super(nvpBatchNorm2d, self).__init__(num_features, eps, momentum, affine,
                                             track_running_stats, device, dtype)
        self._current_var = self.running_var  # safe default before any forward pass

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        lp = torch.log(self.weight) if self.affine else 0
        ildj = lp - 0.5 * torch.log(self._current_var + self.eps)
        return torch.sum(torch.flatten(ildj.view([1, self.num_features, 1, 1]).expand_as(y), 1), 1)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param y:
        :return:
        During training, nn.BatchNorm2d.forward() normalizes each sample using
        statistics computed from the current batch:
            x_norm = (x - mean_batch) / sqrt(var_batch + eps)
        It then updates running_mean and running_var as an exponential moving
        average — but those running stats are not what's used to transform the
        data during training.
        During eval, it switches to using running_mean / running_var
        """
        if self.training:
            # mean over batch and spatial dims, keeping channel dim
            reduce_dims = [0] + list(range(2, y.dim()))  # all dimensions apart from 1, which is the channel
            self._current_var = y.var(reduce_dims, unbiased=False).detach()
        else:
            self._current_var = self.running_var
        return super(nvpBatchNorm2d, self).forward(y), self.forwardLogDetJacobian(y)


class realNVPCouplingLayer(normFlowModule):
    """
    Single NVP coupling layer. Implements the checkerboard and channel
    masks from [Dinh 2016].
    """
    def __init__(self,
                 scaleModule: torch.nn.Module,
                 biasModule: torch.nn.Module,
                 dims: Tuple[int,...],
                 mask: str,
                 flip: Union[int, bool],
                 weightDecay=5e-5,
                 **kwargs):
        super(realNVPCouplingLayer, self).__init__(**kwargs)

        self.s = scaleModule
        self.t = biasModule

        self.sScale = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.tBias = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)
        self.tScale = torch.nn.Parameter(torch.zeros(dims), requires_grad=True)

        self.weightDecay = weightDecay

        if mask == 'check':
            mask = self.checkerBoardMask(dims)
        elif mask == 'channel':
            mask = self.channelMask(dims)
        else:
            raise Exception(f'Unknown masking type: {mask}')
        if flip:
            mask = 1 - mask
        self.register_buffer(name='mask', tensor=mask)

    @staticmethod
    def checkerBoardMask(dims: Tuple[int,...]) -> torch.Tensor:
        return torch.Tensor(1 - np.indices(dims[1:]).sum(axis=0) % 2).unsqueeze(0)

    @staticmethod
    def channelMask(dims: Tuple[int,...]) -> torch.Tensor:
        assert(len(dims) == 3)
        assert(dims[0] % 2 == 0)
        mask = torch.cat([torch.zeros((dims[0] // 2, dims[1], dims[2])),
                          torch.ones((dims[0] // 2, dims[1], dims[2]))], dim=0)
        assert(mask.shape == tuple(dims))
        return mask

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        s = self.sScale * torch.tanh(self.s(self.mask * y))
        return torch.sum(torch.flatten((1 - self.mask) * s, 1), 1)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.tScale * self.t(self.mask * y) + self.tBias
        s = self.sScale * torch.tanh(self.s(self.mask * y))
        out = self.mask * y + (1 - self.mask) * (y * torch.exp(s) + t)
        ll = torch.sum(torch.flatten((1 - self.mask) * s, 1), 1)
        ll = ll - self.weightDecay * torch.sum(self.sScale ** 2) / y.shape[0]
        return out, ll


class realNonVolumePreserving(baseTorchModel):
    def __init__(self, baseBlock, dims, hidden=64, nScales=1, nFinal=4, **kwargs):
        super(realNonVolumePreserving, self).__init__(**kwargs)

        self.checkList = torch.nn.ModuleList()
        self.channelList = torch.nn.ModuleList()
        self.register_buffer('pi', torch.tensor(np.pi))

        for _ in range(nScales):
            self.checkList.append(normFlowSequential(realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 0),
                                                     nvpBatchNorm2d(dims[0], affine=False),
                                                     realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 1),
                                                     nvpBatchNorm2d(dims[0], affine=False),
                                                     realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                          baseBlock(dims[0], hidden),
                                                                          dims, 'check', 0),
                                                     nvpBatchNorm2d(dims[0], affine=False)))

            dims = (4 * dims[0], dims[1] // 2, dims[2] // 2)

            self.channelList.append(normFlowSequential(realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 1),
                                                       nvpBatchNorm2d(dims[0], affine=False),
                                                       realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 0),
                                                       nvpBatchNorm2d(dims[0], affine=False),
                                                       realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                            baseBlock(dims[0], hidden),
                                                                            dims, 'channel', 1),
                                                       nvpBatchNorm2d(dims[0], affine=False)))

            dims = (dims[0] // 2, dims[1], dims[2])
            hidden = hidden * 2

        self.final = normFlowSequential(*[l for i in range(nFinal) for l in
                                          [realNVPCouplingLayer(baseBlock(dims[0], hidden),
                                                                baseBlock(dims[0], hidden),
                                                                dims, 'check', i % 2 == 0),
                                           nvpBatchNorm2d(dims[0], affine=False)]])

    def forward(self, y):
        yr = []
        ll = []

        for check, channel in zip(self.checkList, self.channelList):
            y, ildj = check(y)  # three couplings with checkerboard masking
            ll.append(ildj)

            y = torch.nn.functional.pixel_unshuffle(y, 2)

            y, ildj = channel(y)  # three couplings with channel masking
            ll.append(ildj)

            yr.append(torch.flatten(y[:, y.shape[1] // 2:, :, :], 1))
            y = y[:, :y.shape[1] // 2, :, :]

        y, ildj = self.final(y)
        ll.append(ildj)
        yr.append(torch.flatten(y, 1))

        return torch.flatten(torch.cat(yr, 1), 1), torch.sum(torch.stack(ll, -1), 1)

    def computeLoss(self, data) -> dict:
        X = data.to(self.device)
        y, ll = self.forward(X)

        priorLoss = torch.mean((torch.log(2 * self.pi) + y * y) / 2.)
        logLoss = -torch.mean(ll)
        totalLoss = priorLoss + logLoss

        return {"totalLoss": totalLoss, "logLoss": logLoss, "priorLoss": priorLoss}


#################################
#              MADE             #
#################################
class madeLayer(torch.nn.Linear):
    def __init__(self, inDims: int, outDims: int,
                 bias: bool = True, device=None, dtype=None,
                 index=None, isFinal=False, minIndex=None, maxIndex=None,
                 outDegree: Optional[torch.Tensor] = None) -> None:
        if isFinal:
            super(madeLayer, self).__init__(inDims, 2 * outDims, bias, device, dtype)
        else:
            super(madeLayer, self).__init__(inDims, outDims, bias, device, dtype)

        # set up the mask
        if index is None:
            self.register_buffer("index", torch.randperm(self.in_features))
        else:
            self.register_buffer("index", index)

        if minIndex is None:
            self.minIndex = 0
        else:
            self.minIndex = minIndex

        if maxIndex is None:
            self.maxIndex = self.index.max().item()
        else:
            self.maxIndex = maxIndex

        if isFinal:
            assert outDegree is not None, "outDegree (per-dimension degree assignment) required for final layer"
            outIndex = torch.tile(outDegree, (2, 1)).T.flatten()
            mask = (outIndex.unsqueeze(1) > self.index.unsqueeze(0)).float()
        else:
            outIndex = torch.randint(self.minIndex, self.maxIndex + 1, (outDims,))
            mask = (outIndex.unsqueeze(1) >= self.index.unsqueeze(0)).float()
        self.register_buffer("outIndex", outIndex)
        self.register_buffer("mask", mask)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(y, self.weight * self.mask, self.bias)


class maskedAutoRegressiveFlow(normFlowModule):
    def __init__(self, dims: Iterable[int], activation: Optional[torch.nn.Module] = None):
        super(maskedAutoRegressiveFlow, self).__init__()
        self.madeList = torch.nn.ModuleList()

        self.dim = dims[0]
        self.register_buffer("index", torch.randperm(self.dim))
        self.activation = activation if activation is not None else torch.nn.ReLU()

        if len(dims) == 1:
            self.madeList.append(madeLayer(dims[0], dims[0], bias=True,
                                           index=self.index, isFinal=True,
                                           minIndex=0, maxIndex=self.dim - 1,
                                           outDegree=self.index))
        else:
            self.madeList.append(madeLayer(dims[0], dims[1], bias=True,
                                           index=self.index, isFinal=False,
                                           minIndex=0, maxIndex=self.dim - 1))
            for i in range(1, len(dims) - 1):
                self.madeList.append(madeLayer(dims[i], dims[i + 1], bias=True,
                                               index=self.madeList[-1].outIndex, isFinal=False,
                                               minIndex=0, maxIndex=self.dim - 1))

            self.madeList.append(madeLayer(dims[-1], self.dim, bias=True,
                                           index=self.madeList[-1].outIndex, isFinal=True,
                                           minIndex=0, maxIndex=self.dim - 1,
                                           outDegree=self.index))

    def generate(self, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(eps.shape, device=eps.device)
        for _ in range(self.dim):
            for layer in self.madeList[:-1]:
                x = self.activation(layer(x))
            x = self.madeList[-1](x)  # no relu on the final step
            p = x.reshape(*x.shape[:-1], -1, 2)
            x = p[..., 0] + torch.exp(p[..., 1]) * eps
        return x, torch.sum(p[..., 1], -1)

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = x
        for layer in self.madeList[:-1]:
            p = self.activation(layer(p))
        p = self.madeList[-1](p)  # no relu on the final step!
        p = p.reshape(*p.shape[:-1], -1, 2)
        return (x - p[..., 0]) / torch.exp(p[..., 1]), -torch.sum(p[..., 1], -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.normalize(x)

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        _, lpt = self.forward(y)
        return lpt


class inverseAutoRegressiveFlow(normFlowModule):
    def __init__(self, dims: Iterable[int], activation: Optional[torch.nn.Module] = None):
        super(inverseAutoRegressiveFlow, self).__init__()
        self.madeList = torch.nn.ModuleList()

        self.dim = dims[0]
        self.register_buffer("index", torch.randperm(self.dim))
        self.activation = activation if activation is not None else torch.nn.ReLU()

        if len(dims) == 1:
            self.madeList.append(madeLayer(dims[0], dims[0], bias=True,
                                           index=self.index, isFinal=True,
                                           minIndex=0, maxIndex=self.dim - 1,
                                           outDegree=self.index))
        else:
            self.madeList.append(madeLayer(dims[0], dims[1], bias=True,
                                           index=self.index, isFinal=False,
                                           minIndex=0, maxIndex=self.dim - 1))
            for i in range(1, len(dims) - 1):
                self.madeList.append(madeLayer(dims[i], dims[i + 1], bias=True,
                                               index=self.madeList[-1].outIndex, isFinal=False,
                                               minIndex=0, maxIndex=self.dim - 1))

            self.madeList.append(madeLayer(dims[-1], self.dim, bias=True,
                                           index=self.madeList[-1].outIndex, isFinal=True,
                                           minIndex=0, maxIndex=self.dim - 1,
                                           outDegree=self.index))

    def generate(self, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = eps
        for layer in self.madeList[:-1]:
            p = self.activation(layer(p))
        p = self.madeList[-1](p)  # no relu on the final step!
        p = p.reshape(*p.shape[:-1], -1, 2)
        return p[..., 0] + torch.exp(p[..., 1]) * eps, torch.sum(p[..., 1], -1)

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.zeros(x.shape, device=x.device)
        for _ in range(self.dim):
            for layer in self.madeList[:-1]:
                eps = self.activation(layer(eps))
            eps = self.madeList[-1](eps)  # no relu on the final step
            p = eps.reshape(*eps.shape[:-1], -1, 2)
            eps = (x - p[..., 0]) / torch.exp(p[..., 1])
        return eps, -torch.sum(p[..., 1], -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generate(x)

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        _, lpt = self.forward(y)
        return lpt


#################################
#              CNF              #
#################################
# time embedding layers
class timeEmbedding(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def combine(self, i: int, x: torch.Tensor, layer: torch.nn.Module, t_embed) -> torch.Tensor:
        pass

class identityTimeEmbedding(timeEmbedding):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, 1)   # (B, dim)

    def combine(self, i: int, x: torch.Tensor, layer: torch.nn.Module, t_embed) -> torch.Tensor:
        if i == 0:
            dim = 1 if x.dim() > 2 else -1  # channel axis for spatial h, feature axis for flat h
            return layer(torch.cat([x, t_embed], dim=dim))
        else:
            return layer(x)

class fourierTimeEmbedding(timeEmbedding):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        frequencies = torch.pow(10000.0, -2 * torch.arange(dim // 2) / dim)
        self.register_buffer('frequencies', frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)  # (B, 1) or (1, 1) for a scalar/shared t
        angles = t * self.frequencies  # (B, dim//2,)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)   # (B, dim)

    def combine(self, i: int, x: torch.Tensor, layer: torch.nn.Module, t_embed) -> torch.Tensor:
        if i == 0:
            dim = 1 if x.dim() > 2 else -1  # channel axis for spatial h, feature axis for flat h
            return layer(torch.cat([x, t_embed], dim=dim))
        else:
            return layer(x)

class filmTimeEmbedding(timeEmbedding):
    def __init__(self, dims: List[int]):
        super().__init__()

        # a paired *TimeConditionedNetwork built from `dims` has len(dims)-1
        # FiLM-able layers (all but the final one), with output widths dims[1:]
        self.dims = dims[1:]
        self.net = torch.nn.Linear(1, 2 * sum(self.dims))
        self.dim = 0  # signals to cnfDynamicsFilm: don't add to input dim

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.net(t.reshape(-1, 1)).squeeze(0)  # (2 * sum(dims),)
        chunks = params.split([2 * d for d in self.dims])
        gammas = [c[:d] for c, d in zip(chunks, self.dims)]
        betas = [c[d:] for c, d in zip(chunks, self.dims)]
        return gammas, betas  # gamma, beta: (numLayers, hiddenDim)

    def combine(self, i, x, layer, t_embed):
        gamma, beta = t_embed[i]
        return gamma * layer(x) + beta


# take input and time to produce flow/velocity fields
class scalarConditionedNetwork(torch.nn.Module, abc.ABC):
    """
    Consumes a state tensor z (B x ...) together with an already-computed,
    batch-aligned time embedding t_embed and produces an output of the same
    shape as z. Owns all architecture-specific knowledge — MLP vs conv,
    concatenation vs FiLM, how many layers/blocks — so a timeConditionedField
    itself never needs to know z's shape or how t_embed should combine with it.
    """

    @abc.abstractmethod
    def filmDims(self) -> int:
        pass

    @abc.abstractmethod
    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        pass

class scalarConditionedNetworkMLP(scalarConditionedNetwork):
    """
    Concatenates the time embedding onto z as extra features and runs the
    result through a plain MLP. Assumes z is a flat B x D tensor. t_dim must
    match the dimension of whatever time embedding this is paired with in the
    owning timeConditionedField (1 if none is used).
    """
    def __init__(self, dims: List[int], t_dim: int = 1,
                 conditioning: timeConditioningStrategy = None,
                 activation: Optional[torch.nn.Module] = None):
        super(scalarConditionedNetworkMLP, self).__init__()

        self.conditioning = conditioning or concatConditioning()
        self.activation = activation if activation is not None else torch.nn.ELU()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(dims[0] + self.conditioning.timeDims(t_dim),
                                           dims[1], bias=True))
        for inDims, outDims in zip(dims[1:-1], dims[2:]):
            self.layers.append(torch.nn.Linear(inDims, outDims, bias=True))
        self.layers.append(torch.nn.Linear(dims[-1], dims[0], bias=True))

        self.__filmDims = sum(dims[1:])

    def filmDims(self) -> int:
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(embedding.combine(i, h, layer, t_embed))
        return self.layers[-1](h)

class scalarConditionedNetworkCNN(scalarConditionedNetwork):
    """
    Concatenates the time embedding onto z as extra features and runs the
    result through a plain MLP. Assumes z is a flat B x D tensor. t_dim must
    match the dimension of whatever time embedding this is paired with in the
    owning timeConditionedField (1 if none is used).
    """

    def __init__(self, inDims: int, configs: List[Tuple[int]], t_dim: int = 1,
                 conditioning: timeConditioningStrategy = None,
                 activation: Optional[torch.nn.Module] = None):
        super(scalarConditionedNetworkCNN, self).__init__()

        self.conditioning = conditioning or concatConditioning()
        self.activation = activation if activation is not None else torch.nn.ELU()

        inChannels = inDims + self.conditioning.timeDims(t_dim)
        self.layers = torch.nn.ModuleList()
        for outChannels, kernelSize in configs:
            self.layers.append(torch.nn.Conv2d(inChannels, outChannels, kernelSize,
                                               stride=1, padding='same', bias=True))
            inChannels = outChannels

        # ensure output of same shape as input
        self.layers.append(torch.nn.Conv2d(inChannels, inDims, kernelSize=1, bias=True))

        self.__filmDims = sum([c for c, _, _ in configs])

    def filmDims(self) -> int:
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        t_embed = t_embed.view(*t_embed.shape, 1, 1).expand(-1, -1, *z.shape[2:])  # broadcast to (B, t_dim, H, W)
        h = z
        for layer in self.layers[:-1]:
            h = self.activation(embedding.combine(i, h, layer, t_embed))
        return self.layers[-1](h)

# probes for hutchinson estimator
def gaussianProbe(z: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(z)

def rademacherProbe(z: torch.Tensor) -> torch.Tensor:
    return torch.randint(0, 2, z.shape, device=z.device, dtype=z.dtype) * 2 - 1


class timeConditionedField(torch.nn.Module):
    """
    Pairs a time embedding with a timeConditionedNetwork that consumes z and
    the embedded time together. Owns exactly the time-handling logic that's
    independent of the network's architecture — looking up (or passing
    through) the time embedding, and broadcasting a single shared t (as used
    by a continuousNormFlow's ODE solver, where every row of the batch is at
    the same integration time) across the batch when needed. Everything about
    how z and the embedded time actually combine — concatenation, FiLM,
    convolutional broadcasting, etc. — is the injected network's
    responsibility, letting the same field/hutchinsonTrace/CNF machinery work
    with any network architecture.

    This class can be used for the dynamics of a continuous norm flow or as
    the network that generates the regression target for flow matching and
    score matching.
    """
    def __init__(self, network: scalarConditionedNetwork, embedding=None, probeSampler=None):
        super(timeConditionedField, self).__init__()
        self.network = network
        self.timeEmbedding = embedding if embedding else identityTimeEmbedding()
        self.probeSampler = probeSampler if probeSampler is not None else gaussianProbe

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.timeEmbedding(t)
        if isinstance(t_embed, torch.Tensor) and t_embed.shape[0] == 1:
            # For CFM and diffusion, t will now be of dimension B x 1
            # However, for CNFs, t is constant across the batch and
            # have shape 1 x 1. This needs to be broadcast across the
            # batch so expand if necessary! (FiLM embeddings are unbatched
            # per-layer vectors that already broadcast correctly against
            # z without this step.)
            t_embed = t_embed.expand(z.shape[0], -1)

        return self.network(z, t_embed, self.timeEmbedding)

    def hutchinsonTrace(self, z: torch.Tensor, t: torch.Tensor, f: torch.Tensor = None):
        eps = self.probeSampler(z)
        if f is None:
            f = self.forward(z, t)

        # (df/dz).ε via autograd — compute gradient of d(f·ε)/dz
        jvp = torch.autograd.grad(f, z, grad_outputs=eps, create_graph=True)[0]
        return (eps * jvp).sum(-1)


class continuousNormFlow(normFlowModule):
    def __init__(self, dynamics: timeConditionedField, direction='generate'):
        super(continuousNormFlow, self).__init__()
        self.dynamics = dynamics
        assert direction in ('normalize', 'generate')
        self.direction = direction

    def _integrate(self, y, ts):
        log_p = torch.zeros(y.shape[0], device=y.device)  # initial log det = 0

        def augmentedDynamics(t, state):
            z, lp = state
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                dz_dt = self.dynamics(z, t)
                dlp_dt = -self.dynamics.hutchinsonTrace(z, t, dz_dt)
            return dz_dt, dlp_dt

        zt, lpt = odeint_adjoint(augmentedDynamics, (y, log_p), ts,
                                 method='rk4', options={'step_size': 0.05},
                                 adjoint_params=list(self.dynamics.parameters()))
        return zt[-1], lpt[-1]  # odeint returns values at all t, take the final

    def generate(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._integrate(y, torch.tensor([0., 1.], device=y.device))

    def normalize(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, lpt = self._integrate(y, torch.tensor([1., 0.], device=y.device))
        return z, -lpt  # reversed-time trace integral is the forward map's logdet, negate for z→u0

    def forward(self, y):
        if self.direction == 'normalize':
            return self.normalize(y)
        else:
            return self.generate(y)

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        _, lpt = self.forward(y)
        return lpt


if __name__ == "__main__":
    m = nvpBatchNorm2d(100)
    input = torch.randn(20, 100, 35, 45)
    output = m(input)
    output = m.forwardLogDetJacobian(input)

    shape = (1, 28, 28)
    planes = 64
    for k in range(6):
        print(k, shape)
        if k % 6 == 2:
            shape = (4 * shape[0], shape[1] // 2, shape[2] // 2)
