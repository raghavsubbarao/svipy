import abc
import math
import numpy as np

from typing import Optional, Iterable, Union, overload, Tuple, List

import torch
from torchdiffeq import odeint_adjoint, odeint

from svipy.model import baseTorchModel
from svipy.recurrent.attention import multiHeadAttentionTorchSDP

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
    def __init__(self, flow: normFlowModule):
        self.flow = flow

    @abc.abstractmethod
    def baseLikelihood(self, z: torch.Tensor) -> torch.Tensor:
        pass

    def logLikelihood(self, z: torch.Tensor) -> torch.Tensor:
        u, logDet = self.flow(z)
        logBase = self.baseLikelihood(u)
        return logBase + logDet

class normFlowPriorUniform(normFlowPrior):
    def __init__(self, flow: normFlowModule):
        super(normFlowPriorUniform, self).__init__(flow)

    def baseLikelihood(self, u):
        return 0

class normFlowPriorNormal(normFlowPrior):
    def __init__(self, flow: normFlowModule):
        super(normFlowPriorNormal, self).__init__(flow)

    def baseLikelihood(self, u):
        dim = u.flatten(start_dim=1).shape[1]
        return -0.5 * (u.pow(2).flatten(start_dim=1).sum(dim=1) + dim * math.log(2 * math.pi))


class normFlowPosterior:
    def __init__(self, flow: normFlowModule):
        self.flow = flow
        # self.dim = dim

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

    @staticmethod
    def _expandToMatch(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        v: (leadingDim, k) - a per-example (or shared, leadingDim=1) vector.
        Pads trailing singleton dims and expands so it broadcasts against a
        spatial x of shape (B, k, *spatial); a no-op for flat x (B, k).
        """
        if x.dim() > 2:
            v = v.view(*v.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:])
        return v

class identityTimeEmbedding(timeEmbedding):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, 1)   # (B, dim)

    def combine(self, i: int, x: torch.Tensor, layer: torch.nn.Module, t_embed) -> torch.Tensor:
        if i == 0:
            dim = 1 if x.dim() > 2 else -1  # channel axis for spatial x, feature axis for flat x
            t_embed = self._expandToMatch(t_embed, x)
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
            dim = 1 if x.dim() > 2 else -1  # channel axis for spatial x, feature axis for flat x
            t_embed = self._expandToMatch(t_embed, x)
            return layer(torch.cat([x, t_embed], dim=dim))
        else:
            return layer(x)

class filmTimeEmbedding(timeEmbedding):
    def __init__(self, nFrequencies: int, outDims: List[int],
                 hiddenDims: Optional[List[int]] = None,
                 activation: Optional[torch.nn.Module] = None):
        # dims: the output width of every FiLM-able layer, in order -
        # i.e. exactly network.filmDims() for whichever network this pairs
        # with (scalarConditionedNetworkMLP/CNN). One (gamma, beta) pair gets
        # produced per entry, sized to that entry's width.
        super().__init__()
        assert nFrequencies % 2 == 0
        self.nFrequencies = nFrequencies
        self.dims = outDims

        frequencies = torch.pow(10000.0, -2 * torch.arange(nFrequencies // 2) / nFrequencies)
        self.register_buffer('frequencies', frequencies)

        self.activation = activation if activation is not None else torch.nn.SiLU()

        if hiddenDims is None:
            self.net = torch.nn.ModuleList()
            self.out = torch.nn.Linear(nFrequencies, 2 * sum(self.dims))
        else:
            dims = [nFrequencies] + hiddenDims
            self.net = torch.nn.ModuleList([torch.nn.Linear(d_in, d_out)
                                            for d_in, d_out in zip(dims[:-1], dims[1:])])
            self.out = torch.nn.Linear(hiddenDims[-1], 2 * sum(self.dims))

        self.dim = 0  # signals to cnfDynamicsFilm: don't add to input dim

    def forward(self, t: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        t = t.reshape(-1, 1)  # (B, 1) or (1, 1) for a scalar/shared t
        angles = t * self.frequencies  # (B, dim//2)
        h = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
        for layer in self.net:
            h = self.activation(layer(h))  # (leadingDim, 2 * sum(dims))
        params = self.out(h)  # no activation on final step
        chunks = params.split([2 * d for d in self.dims], dim=-1)
        gammas = [c[..., :d] for c, d in zip(chunks, self.dims)]
        betas = [c[..., d:] for c, d in zip(chunks, self.dims)]
        return list(zip(gammas, betas))  # one (gamma_i, beta_i) pair per layer, indexable by i

    def combine(self, i, x, layer, t_embed):
        gamma, beta = t_embed[i]  # each (leadingDim, d_i)
        out = layer(x)
        gamma = self._expandToMatch(gamma, out)
        beta = self._expandToMatch(beta, out)
        return gamma * out + beta

class adaGnTimeEmbedding(filmTimeEmbedding):
    def __init__(self, nFrequencies: int, outDims: List[int],
                 hiddenDims: Optional[List[int]] = None,
                 activation: Optional[torch.nn.Module] = None):
        super().__init__(adaGnTimeEmbedding, self).__init__(nFrequencies, outDims,
                                                            hiddenDims, activation)

    def combine(self, i, x, layer, t_embed):
        gamma, beta = t_embed[i]  # each (leadingDim, d_i)
        out = layer(x)
        gamma = self._expandToMatch(gamma, out)
        beta = self._expandToMatch(beta, out)
        return (1 + gamma) * out + beta

# take input and time to produce flow/velocity fields
class scalarConditionedNetwork(torch.nn.Module, abc.ABC):
    """
    Consumes a state tensor z (B x ...) together with an already-computed,
    batch-aligned time embedding t_embed and produces an output of the same
    shape as z. Owns all architecture-specific knowledge — MLP vs conv,
    how many layers/blocks — while the *embedding* now owns how t_embed
    combines with each layer (concatenation, FiLM, ...), since that combine
    logic depends on the embedding, not the backbone.
    """
    @abc.abstractmethod
    def filmDims(self) -> List[int]:
        """
        Output width of every FiLM-able layer, in order - what a paired
        filmTimeEmbedding needs to produce one (gamma, beta) pair per entry.
        """
        pass

    @abc.abstractmethod
    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        pass

class scalarConditionedNetworkMLP(scalarConditionedNetwork):
    """
    A plain MLP. t_dim is the extra input width the first layer needs for a
    concatenating embedding (identity/Fourier); pass t_dim=0 if this will
    only ever be paired with a FiLM embedding, which never concatenates.
    filmDims() reports the per-layer widths a filmTimeEmbedding needs to
    produce to modulate every layer of this network.
    """
    def __init__(self, dims: List[int], t_dim: int = 1,
                 activation: Optional[torch.nn.Module] = None):
        super(scalarConditionedNetworkMLP, self).__init__()

        self.activation = activation if activation is not None else torch.nn.ELU()

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(dims[0] + t_dim, dims[1], bias=True))
        for inDims, outDims in zip(dims[1:-1], dims[2:]):
            self.layers.append(torch.nn.Linear(inDims, outDims, bias=True))
        self.layers.append(torch.nn.Linear(dims[-1], dims[0], bias=True))

        self.__t_dim = t_dim
        self.__filmDims = dims[1:]

    def filmDims(self) -> List[int]:
        assert self.__t_dim == 0, (f"filmDims() is only meaningful for a network built with t_dim=0 "
                                   f"(FiLM never concatenates); this one was built with t_dim={self.__t_dim}")
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(embedding.combine(i, h, layer, t_embed))
        return self.layers[-1](h)

class scalarConditionedNetworkCNN(scalarConditionedNetwork):
    """
    Runs z through a stack of stride-1, same-padding convolutions. Every conv
    preserves spatial shape, and the final 1x1 conv always maps back to
    inDims channels, so the output shape matches the input shape by
    construction rather than needing a runtime check. t_dim/filmDims() follow
    the same convention as scalarConditionedNetworkMLP.
    """

    def __init__(self, inDims: int, configs: List[Tuple[int, int]], t_dim: int = 1,
                 activation: Optional[torch.nn.Module] = None):
        super(scalarConditionedNetworkCNN, self).__init__()

        self.activation = activation if activation is not None else torch.nn.ELU()

        inChannels = inDims + t_dim
        self.layers = torch.nn.ModuleList()
        for outChannels, kernelSize in configs:
            self.layers.append(torch.nn.Conv2d(inChannels, outChannels, kernelSize,
                                               stride=1, padding='same', bias=True))
            inChannels = outChannels

        # ensure output of same shape as input
        self.layers.append(torch.nn.Conv2d(inChannels, inDims, kernel_size=1, bias=True))

        self.__t_dim = t_dim
        self.__filmDims = [outChannels for outChannels, _ in configs]

    def filmDims(self) -> List[int]:
        assert self.__t_dim == 0, (f"filmDims() is only meaningful for a network built with t_dim=0 "
                                   f"(FiLM never concatenates); this one was built with t_dim={self.__t_dim}")
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(embedding.combine(i, h, layer, t_embed))
        return self.layers[-1](h)

class scalarConditionedNetworkUNet(scalarConditionedNetwork):
    """
    A small U-Net: len(channels) downsampling stages (each a same-padding
    conv then a stride-2 conv to halve resolution), a bottleneck, then
    len(channels) upsampling stages (each a stride-2 transpose conv then a
    same-padding conv) that concatenate the matching downsampling stage's
    pre-downsample activation as a skip connection. The last upsampling
    stage maps back to inDims, undoing the first downsampling stage exactly,
    so there's no separate final projection layer.

    Meant to pair with filmTimeEmbedding (t_dim=0) - FiLM is applied to
    every conv's output (every down stage, the bottleneck, every up stage),
    matching how real diffusion U-Nets inject time at every resolution.

    Input spatial size must be divisible by 2**len(channels) (each down
    stage halves it; the up path needs to land back on an integer size).
    """
    def __init__(self, inDims: int, channels: List[int], kernelSize: int = 3,
                 t_dim: int = 1, activation: Optional[torch.nn.Module] = None,
                 addResidual: bool = False, addAttention: bool = False):
        super(scalarConditionedNetworkUNet, self).__init__()
        self.activation = activation if activation is not None else torch.nn.ELU()

        dims = [inDims + t_dim] + list(channels)
        nStages = len(channels)

        self.downs = torch.nn.ModuleList([torch.nn.Conv2d(dims[i], dims[i + 1], kernelSize, padding='same')
                                          for i in range(nStages)])
        self.downsamples = torch.nn.ModuleList([torch.nn.Conv2d(dims[i + 1], dims[i + 1], 4, stride=2, padding=1)
                                                for i in range(nStages)])

        if addResidual:
            self.residuals = torch.nn.ModuleList([torch.nn.Conv2d(dims[i + 1], dims[i + 1], kernelSize, padding='same')
                                                  for i in range(nStages)])
        else:
            self.residuals = None

        self.bottleneck = torch.nn.Conv2d(dims[-1], dims[-1], kernelSize, padding='same')

        if addAttention:
            # dims has nStages+1 entries (0..nStages) - the bottleneck width is
            # dims[nStages] == dims[-1], same value self.bottleneck itself uses.
            self.sha = multiHeadAttentionTorchSDP(dims[nStages], dims[nStages], 4,
                                                  bias=True, is_causal=False)
            # pre-norm before the attention residual, standard transformer-block
            # practice - without it the un-normalized bottleneck activations
            # (already FiLM-scaled, arbitrary magnitude) feed straight into the
            # softmax, which can make attention poorly calibrated and adds
            # another unnormalized residual stack on top of the conv path -
            # exactly the kind of thing that destabilized training before
            # gradient clipping was added.
            self.attNorm = torch.nn.LayerNorm(dims[nStages])
        else:
            self.sha = None
            self.attNorm = None

        # up stage j undoes down stage (nStages-1-j): upsamples to dims[nStages-j]
        # (matching the skip from that stage for concatenation), then the fuse conv
        # maps the concatenated 2*dims[nStages-j] channels down to dims[nStages-1-j]
        self.upsamples = torch.nn.ModuleList([torch.nn.ConvTranspose2d(dims[nStages - j], dims[nStages - j],
                                                                       4, stride=2, padding=1)
                                              for j in range(nStages)])
        self.ups = torch.nn.ModuleList([torch.nn.Conv2d(dims[nStages - j] * 2, dims[nStages - 1 - j],
                                                        kernelSize, padding='same')
                                        for j in range(nStages)])

        # built to mirror the construction loops above exactly
        downWidths = []
        for i in range(nStages):
            downWidths.append(dims[i + 1])
            if addResidual:
                downWidths.append(dims[i + 1])
        bottleneckWidth = [dims[-1]]
        upWidths = [dims[nStages - 1 - j] for j in range(nStages)]
        self.__filmDims = downWidths + bottleneckWidth + upWidths

    def filmDims(self) -> List[int]:
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        skips = []

        x = z
        i = 0
        for j, (down, downsample) in enumerate(zip(self.downs, self.downsamples)):
            x = self.activation(embedding.combine(i, x, down, t_embed))
            i = i + 1
            if self.residuals:
                x = x + self.activation(embedding.combine(i, x, self.residuals[j], t_embed))
                i = i + 1
            skips.append(x)
            x = downsample(x)

        x = self.activation(embedding.combine(i, x, self.bottleneck, t_embed))
        if self.sha:
            dims = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            tokens = tokens + self.sha(self.attNorm(tokens))
            x = tokens.transpose(1, 2).reshape(dims)
        i = i + 1

        for j, (upsample, up) in enumerate(zip(self.upsamples, self.ups)):
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = embedding.combine(i, x, up, t_embed)
            if j < len(self.ups) - 1:  # no activation on the final output layer, matching
                x = self.activation(x)  # scalarConditionedNetworkCNN's own convention
            i = i + 1

        return x


class UNetResidualBlock(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, kernelSize=3, c_norm=32, activation: Optional[torch.nn.Module] = None):
        super(UNetResidualBlock, self).__init__()
        self.c_norm = c_norm
        # GroupNorm needs num_channels divisible by num_groups, which the
        # first down stage's c_in (inDims+t_dim, e.g. 3 for RGB) generally
        # isn't for any reasonable c_norm - clamp down to c_in itself in
        # that case (degrades to per-channel/InstanceNorm-like behavior
        # only for that one layer; every other stage's width is large
        # enough that this is a no-op).
        self.norm1 = torch.nn.GroupNorm(c_norm, c_in)
        self.conv1 = torch.nn.Conv2d(c_in, c_out, kernelSize, padding='same')
        self.norm2 = torch.nn.GroupNorm(c_norm, c_out)
        self.conv2 = torch.nn.Conv2d(c_out, c_out, kernelSize, padding='same')
        self.skip = torch.nn.Conv2d(c_in, c_out, 1) if c_in != c_out else torch.nn.Identity()
        self.activation = activation if activation is not None else torch.nn.SiLU()

        torch.nn.init.zeros_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding, i: int):
        # keep x as the original, untouched block input throughout - self.skip(x)
        # below needs it at its original c_in width, so the transformed path has
        # to live in its own variable rather than reusing/overwriting x. self.skip
        # is always a module (Conv2d or Identity - never None), so no if/else needed.
        h = self.conv1(self.activation(self.norm1(x)))
        h = embedding.combine(i, h, self.norm2, t_embed)
        return self.skip(x) + self.conv2(self.activation(h))

class convolutionalAttention(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, nHeads: int):
        super(convolutionalAttention, self).__init__()
        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=c_in)
        self.mha = multiHeadAttentionTorchSDP(c_in, c_out, nHeads, bias=True, is_causal=False)

        torch.nn.init.zeros_(self.mha.ff.weight)
        torch.nn.init.zeros_(self.mha.ff.bias)

    def forward(self, x: torch.Tensor):
        # x is of shape B x C x H X W so reshape to apply attention
        dims = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        return x + self.mha(tokens).transpose(1, 2).reshape(dims)

class scalarConditionedNetworkUNetResidual(scalarConditionedNetwork):
    """
    A GroupNorm/AdaGN residual U-Net, closer to the canonical DDPM/ADM
    architecture than scalarConditionedNetworkUNet: an initial stem
    convolution (embed) lifts the input to channels[0] at full resolution,
    then len(channels)-1 downsampling stages (each a GroupNorm residual
    block then a stride-2 conv to halve resolution), a bottleneck (residual
    block, self-attention, residual block), then len(channels)-1 upsampling
    stages (stride-2 transpose conv then a residual block) that concatenate
    the matching downsampling stage's pre-downsample activation and its
    post-downsample activation as two skip connections. A final
    GroupNorm+activation+zero-initialized conv projects back from
    channels[0] to inDims.

    Meant to pair with filmTimeEmbedding (t_dim=0) - FiLM is applied to
    every residual block's second GroupNorm (AdaGN), matching how real
    diffusion U-Nets inject time.

    Input spatial size must be divisible by 2**(len(channels)-1) (each
    down stage halves it; the up path needs to land back on an integer
    size).
    """
    def __init__(self, inDims: int, channels: List[int], kernelSize: int = 3,
                 t_dim: int = 1, activation: Optional[torch.nn.Module] = None):
        super(scalarConditionedNetworkUNetResidual, self).__init__()
        self.activation = activation if activation is not None else torch.nn.SiLU()
        dims = list(channels)
        nStages = len(channels) - 1

        self.embed = torch.nn.Conv2d(inDims + t_dim, channels[0], 3, padding=1)
        self.norm = torch.nn.GroupNorm(min(8, channels[0]), channels[0])
        self.unembed = torch.nn.Conv2d(channels[0], inDims, kernelSize, padding='same')
        torch.nn.init.zeros_(self.unembed.weight)
        torch.nn.init.zeros_(self.unembed.bias)

        # encoder
        self.downs = torch.nn.ModuleList([UNetResidualBlock(dims[i], dims[i + 1],
                                                            kernelSize=kernelSize,
                                                            c_norm=8, activation=self.activation)
                                          for i in range(nStages)])
        self.downsamples = torch.nn.ModuleList([torch.nn.Conv2d(dims[i + 1], dims[i + 1],
                                                                kernel_size=4, stride=2,
                                                                padding=1)
                                                for i in range(nStages)])

        # bottleneck
        self.bottleneck = torch.nn.ModuleList([UNetResidualBlock(dims[-1], dims[-1],
                                                                 kernelSize=kernelSize,
                                                                 c_norm=8, activation=self.activation),
                                               convolutionalAttention(dims[-1], dims[-1], 4),
                                               UNetResidualBlock(dims[-1], dims[-1],
                                                                 kernelSize=kernelSize,
                                                                 c_norm=8, activation=self.activation)])

        # up stage j undoes down stage (nStages-1-j): upsamples to dims[nStages-j]
        # (matching the skip from that stage for concatenation), then the fuse conv
        # maps the concatenated 2*dims[nStages-j] channels down to dims[nStages-1-j]
        self.upsamples = torch.nn.ModuleList([torch.nn.ConvTranspose2d(dims[i] * 2, dims[i],
                                                                       kernel_size=4,
                                                                       stride=2, padding=1)
                                              for i in range(nStages, 0, -1)])
        self.ups = torch.nn.ModuleList([UNetResidualBlock(dims[i] * 2, dims[i - 1],
                                                          kernelSize=kernelSize,
                                                          c_norm=8, activation=self.activation)
                                        for i in range(nStages, 0, -1)])

        # built to mirror the construction loops above exactly - one entry
        # per combine() call in forward(). downs[i]'s combine() call acts on
        # conv1's output (norm2 is GroupNorm(c_out)), so the width needed is
        # dims[i+1], not dims[i] (dims[i] is downs[i]'s *input* width).
        # Nothing is entered for the attention block in the bottleneck since
        # it never calls combine() - forward() correspondingly never
        # advances i for it either.
        downWidths = [dims[j + 1] for j in range(nStages)]
        bottleneckWidth = [dims[-1], dims[-1]]
        upWidths = [dims[nStages - 1 - j] for j in range(nStages)]
        self.__filmDims = downWidths + bottleneckWidth + upWidths

    def filmDims(self) -> List[int]:
        return self.__filmDims

    def forward(self, z: torch.Tensor, t_embed: torch.Tensor, embedding: timeEmbedding) -> torch.Tensor:
        skips = []

        x = self.embed(z)
        i = 0
        for j, (down, downsample) in enumerate(zip(self.downs, self.downsamples)):
            x = down(x, t_embed, embedding, i)
            skips.append(x)
            i = i + 1

            x = downsample(x)
            skips.append(x)

        # res block
        x = self.bottleneck[0](x, t_embed, embedding, i)
        i = i + 1

        # attention block
        x = self.bottleneck[1](x)

        # res block
        x = self.bottleneck[2](x, t_embed, embedding, i)
        i = i + 1

        for j, (upsample, up) in enumerate(zip(self.upsamples, self.ups)):
            x = upsample(torch.cat([x, skips.pop()], dim=1))
            x = up(torch.cat([x, skips.pop()], dim=1), t_embed, embedding, i)
            i = i + 1
            if j < len(self.ups) - 1:  # no activation on the final output layer, matching
                x = self.activation(x)  # scalarConditionedNetworkCNN's own convention

        return self.unembed(self.activation(self.norm(x)))


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
            # For CFM and diffusion, t will now be of dimension B x ...
            # However, for CNFs, t is constant across the batch and will have
            # shape 1 x 1. This needs to be broadcast across the batch so expand
            # if necessary! (FiLM embeddings are unbatched per-layer vectors
            # that already broadcast correctly against z without this step.)
            t_embed = t_embed.expand(z.shape[0], -1)

        return self.network(z, t_embed, self.timeEmbedding)

    def hutchinsonTrace(self, z: torch.Tensor, t: torch.Tensor, f: torch.Tensor = None):
        eps = self.probeSampler(z)
        if f is None:
            f = self.forward(z, t)

        # (df/dz).ε via autograd — compute gradient of d(f·ε)/dz
        jvp = torch.autograd.grad(f, z, grad_outputs=eps, create_graph=True)[0]
        return (eps * jvp).flatten(start_dim=1).sum(-1)  # sum over every non-batch dim, not just the last


class continuousNormFlow(normFlowModule):
    def __init__(self, dynamics: timeConditionedField, direction='generate',
                 odeMethod='rk4', odeOptions={'step_size': 0.05}):
        super(continuousNormFlow, self).__init__()
        self.dynamics = dynamics
        assert direction in ('normalize', 'generate')
        self.direction = direction
        self.odeMethod = odeMethod
        self.odeOptions = odeOptions

        # kinetic energy of the last integration, ∫||f(z,t)||^2 dt per example
        # - a diagnostic/regularization quantity (Finlay et al. 2020), updated
        # on every _integrate() call that tracks it. Kept as a side-channel``
        # attribute rather than a return value so this class still satisfies
        # normFlowModule's shared (transformed, logDet) signature like every
        # other flow type; anyone holding a continuousNormFlow instance -
        # whether it's wrapped as a prior or a posterior - can read it after
        # calling forward/generate/normalize. None after a call that opts
        # out of tracking it (interpolate()), rather than silently holding a
        # stale value from a previous generate()/normalize() call.
        self.kineticEnergy = None

    def _integrate(self, y, ts, useAdjoint=True, trackKineticEnergy=True):
        """
        useAdjoint=True uses odeint_adjoint's O(1)-memory continuous adjoint
        method vs plain odeint. Only relevant when something outside this
        call needs to backprop thru the integration into self.dynamics.parameters()
        (e.g. generate()/normalize() used inside a VAE's training loss)
        - pass False for pure sampling/inference use (e.g. interpolate() called
        under torch.no_grad()), where odeint avoids the adjoint bookkeeping cost.

        trackKineticEnergy adds a third augmented ODE state (see self.kineticEnergy
        above) that the solver has to carry and step alongside z and the log-det -
        a cost for callers that never read it, so interpolate() opts out.

        Neither flag affects the enable_grad/requires_grad_ below - that's required
        by hutchinsonTrace's internal torch.autograd.grad call, which needs z to
        require grad at the point dz_dt is computed regardless of solver choice or
        whether the caller wants gradients afterward. Dropping it breaks the log-det
        estimate outright, even with useAdjoint=False.
        """
        log_p = torch.zeros(y.shape[0], device=y.device)  # initial log det = 0
        state = (y, log_p, torch.zeros(y.shape[0], device=y.device)) if trackKineticEnergy else (y, log_p)

        def augmentedDynamics(t, s):
            with torch.enable_grad():
                z = s[0].detach().requires_grad_(True)
                dz_dt = self.dynamics(z, t)
                dlp_dt = -self.dynamics.hutchinsonTrace(z, t, dz_dt)
                if trackKineticEnergy:
                    return dz_dt, dlp_dt, dz_dt.flatten(start_dim=1).pow(2).sum(dim=1)
                return dz_dt, dlp_dt

        if useAdjoint:
            result = odeint_adjoint(augmentedDynamics, state, ts,
                                    method=self.odeMethod, options=self.odeOptions,
                                    adjoint_params=list(self.dynamics.parameters()))
        else:
            result = odeint(augmentedDynamics, state, ts,
                            method=self.odeMethod, options=self.odeOptions)

        if trackKineticEnergy:
            zt, lpt, et = result
            self.kineticEnergy = et[-1]
        else:
            zt, lpt = result
            self.kineticEnergy = None

        return zt[-1], lpt[-1]  # odeint returns values at all t, take the final

    def interpolate(self, y: torch.Tensor, t0: float, t1: float) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._integrate(y, torch.tensor([t0, t1], device=y.device),
                               useAdjoint=False, trackKineticEnergy=False)

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
