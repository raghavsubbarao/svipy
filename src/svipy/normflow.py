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
                 index=None, isFinal=False, minIndex=None, maxIndex=None) -> None:
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
            outIndex = torch.tile(torch.arange(0, outDims), (2, 1)).T.flatten()
            mask = (outIndex.unsqueeze(1) > self.index.unsqueeze(0)).float()
        else:
            outIndex = torch.randint(self.minIndex, self.maxIndex + 1, (outDims,))
            mask = (outIndex.unsqueeze(1) >= self.index.unsqueeze(0)).float()
        self.register_buffer("outIndex", outIndex)
        self.register_buffer("mask", mask)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(y, self.weight * self.mask, self.bias)


class maskedAutoRegressiveFlow(normFlowModule):
    def __init__(self, dims: Iterable[int]):
        super(maskedAutoRegressiveFlow, self).__init__()
        self.madeList = torch.nn.ModuleList()

        self.dim = dims[0]
        self.register_buffer("index", torch.randperm(self.dim))

        if len(dims) == 1:
            pass
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
                                           minIndex=0, maxIndex=self.dim - 1))

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        _, lpt = self.forward(y)
        return lpt

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(y.shape, device=y.device)
        for _ in range(self.dim):
            for layer in self.madeList:
                z = layer(z)
            p = z.reshape(z.shape[0], -1, 2)
            z = p[:, :, 0] + torch.exp(p[:, :, 1]) * y
        return z, torch.sum(p[:, :, 1], -1)


class inverseAutoRegressiveFlow(normFlowModule):
    def __init__(self, dims: Iterable[int]):
        super(inverseAutoRegressiveFlow, self).__init__()
        self.madeList = torch.nn.ModuleList()

        self.dim = dims[0]
        self.register_buffer("index", torch.randperm(self.dim))

        if len(dims) == 1:
            pass
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
                                           minIndex=0, maxIndex=self.dim - 1))

    def forwardLogDetJacobian(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        _, lpt = self.forward(y)
        return lpt

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = y
        for layer in self.madeList:
            p = layer(p)
        p = p.reshape(p.shape[0], -1, 2)
        return p[:,:,0] + torch.exp(p[:,:,1]) * y, torch.sum(p[:,:,1], -1)


#################################
#              CNF              #
#################################
class fourierTimeEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        frequencies = torch.pow(10000.0, -2 * torch.arange(dim // 2) / dim)
        self.register_buffer('frequencies', frequencies)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t * self.frequencies                                      # (dim//2,)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)   # (dim,)


class filmTimeEmbedding(torch.nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims[1:-1]
        self.net = torch.nn.Linear(1, 2 * sum(self.dims))
        self.dim = 0  # signals to cnfDynamicsFilm: don't add to input dim

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.net(t.reshape(1)).squeeze(0)  # (2 * sum(dims),)
        chunks = params.split([2 * d for d in self.dims])
        gammas = [c[:d] for c, d in zip(chunks, self.dims)]
        betas = [c[d:] for c, d in zip(chunks, self.dims)]
        return gammas, betas  # gamma, beta: (numLayers, hiddenDim)


class timeConditionedField(torch.nn.Module):
    """
    A timeConditionedField is a network that takes two inputs - a tensor
    and a scalar 'time' variable. The time variable is treated as an additional
    input, or can be passed through a time embedding network to handle more
    complex dynamics.

    This class can be sued for the dynamics of a continuous norm flow or as
    the network that generates the regression target for flow matching and
    score matching.
    """
    def __init__(self, dims, timeEmbedding=None, _allowFilm=False):
        if timeEmbedding is not None and isinstance(timeEmbedding, filmTimeEmbedding) and not _allowFilm:
            # timeConditionedField cannot handle FiLM time embeddings. However, the timeConditionedFieldFilm
            # initializer also passes through here so in that case _allowFilm=True, but if not throw an error
            raise ValueError("filmTimeEmbedding cannot be used with timeConditionedField — use timeConditionedFieldFilm instead")

        super(timeConditionedField, self).__init__()

        self.timeEmbedding = timeEmbedding
        t_dim = timeEmbedding.dim if timeEmbedding else 1

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(dims[0] + t_dim, dims[1], bias=True))
        for inDims, outDims in zip(dims[1:-1], dims[2:]):
            self.layers.append(torch.nn.Linear(inDims, outDims, bias=True))
        self.layers.append(torch.nn.Linear(dims[-1], dims[0], bias=True))
        self.activation = torch.nn.ELU()

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.timeEmbedding(t) if self.timeEmbedding else t.reshape(1)
        t_expand = t_embed.unsqueeze(0).expand(z.shape[0], -1)
        h = torch.cat([z, t_expand], dim=-1)
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        return self.layers[-1](h)

    def hutchinsonTrace(self, z: torch.Tensor, t: torch.Tensor, f: torch.Tensor = None):
        # todo: implement rademacher sampling (as opposed to normal)
        # todo: sample multiple esp to reduce variance?
        eps = torch.randn_like(z)
        if f is None:
            f = self.forward(z, t)

        # (df/dz).ε via autograd — compute gradient of d(f·ε)/dz
        jvp = torch.autograd.grad(f, z, grad_outputs=eps, create_graph=False)[0]
        return (eps * jvp).sum(-1)


class timeConditionedFieldFilm(timeConditionedField):
    def __init__(self, dims: List[int], film: filmTimeEmbedding):
        super(timeConditionedFieldFilm, self).__init__(dims, film, _allowFilm=True)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.timeEmbedding(t)
        h = z
        for i, layer in enumerate(self.layers[:-1]):
            h = self.activation(gamma[i] * layer(h) + beta[i])
        return self.layers[-1](h)


class continuousNormFlow(normFlowModule):
    def __init__(self, dynamics: timeConditionedField):
        super(continuousNormFlow, self).__init__()
        self.dynamics = dynamics

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_p = torch.zeros(y.shape[0], device=y.device)  # initial log det = 0

        def augmentedDynamics(t, state):
            z, lp = state
            z = z.requires_grad_(True)
            dz_dt = self.dynamics(z, t)
            dlp_dt = -self.dynamics.hutchinsonTrace(z, t, dz_dt)
            return dz_dt, dlp_dt

        ts = torch.tensor([0.0, 1.0], device=y.device)
        zt, lpt = odeint_adjoint(augmentedDynamics, (y, log_p), ts, method='dopri5',
                                 adjoint_params=list(self.dynamics.parameters()))
        return zt[-1], lpt[-1]  # odeint returns values at all t, take the final

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
