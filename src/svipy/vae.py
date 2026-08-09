import abc
from typing import Optional, Iterable, Union, overload, Tuple

import torch

from mlPlayGround.model import baseTorchModel, baseLossTracker

class vaeEncoder:
    def __init__(self, nn: torch.nn.Module) -> None:
        self.__module = nn

    def __getattr__(self, item):
        return getattr(self.__module, item)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.__module(x)

    @property
    def module(self):
        return self.__module

    @staticmethod
    def sampleLatent(mu: torch.Tensor, logVar: torch.Tensor) -> torch.Tensor:
        """
        :param mu:
        :param logVar:
        :return:
        """
        samples = torch.normal(torch.zeros(mu.shape), torch.ones(mu.shape)).to(mu.device)
        return mu + torch.exp(logVar / 2.) * samples

    @staticmethod
    def klLoss(mu: torch.Tensor, logVar: torch.Tensor) -> torch.Tensor:
        """
        :param mu:
        :param logVar:
        :return:
        """
        return torch.mean(torch.sum((torch.exp(logVar) + torch.square(mu) - logVar - mu.shape[1]) / 2., dim=1))

class vaeDecoder:
    def __init__(self, nn: torch.nn.Module) -> None:
        self.__module = nn

    def __getattr__(self, item):
        return getattr(self.__module, item)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.__module(x)

    @property
    def module(self):
        return self.__module

    @staticmethod
    def reconstructionLoss(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :param recon:
        :return:
        """
        return torch.nn.functional.mse_loss(x, recon, reduction='mean')


#################################
#               VAE             #
#################################
class variationalAutoencoder(baseTorchModel):
    """
    Variational Autoencoder implementation. Defaults to \beta=1
    but can also act as a beta VAE
    """
    def __init__(self, encoder: vaeEncoder, decoder: vaeDecoder, beta: float = 1.0, **kwargs):
        # initialization
        super(variationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.register_module('encoder_module', self.encoder.module)
        self.register_module('decoder_module', self.decoder.module)

        # beta = 1.0 for the original [Kingma,Welling 2013] version but can
        # be adjusted for beta-VAEs.
        # beta > 1 allows for disentangling of generative factors [Higgins 2017]
        self.beta = beta

        self.totalLossTracker = baseLossTracker(name="total_loss")
        self.reconLossTracker = baseLossTracker(name="recon_loss")
        self.klLossTracker = baseLossTracker(name="kl_loss")

    @property
    def metrics(self):
        return [self.totalLossTracker,
                self.reconLossTracker,
                self.klLossTracker]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logVar = torch.split(self.encoder(x), split_size_or_sections=2, dim=1)
        return mean, logVar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zMean, zLogVar = self.encode(inputs)  # get the mean and variance
        recon = self.decoder(self.encoder.sampleLatent(zMean, zLogVar))
        return recon, zMean, zLogVar

    def trainStep(self, data, optimizer):
        # implement the abstract method for a single training update
        X = data.to(self.device)
        recon, zMean, zLogVar = self.forward(X)

        reconLoss = self.decoder.reconstructionLoss(X, recon)
        klLoss = self.encoder.klLoss(zMean, zLogVar)
        totalLoss = reconLoss + self.beta * klLoss

        # Backpropagation
        totalLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.totalLossTracker.updateState(totalLoss)
        self.reconLossTracker.updateState(reconLoss)
        self.klLossTracker.updateState(klLoss)

        return {"totalLoss": self.totalLossTracker.result(),
                "reconLoss": self.reconLossTracker.result(),
                "klLoss": self.klLossTracker.result()}

    def validStep(self, data):
        # implement the abstract method for a single training update
        X = data.to(self.device)
        recon, zMean, zLogVar = self.forward(X)

        reconLoss = self.decoder.reconstructionLoss(X, recon)
        klLoss = self.encoder.klLoss(zMean, zLogVar)
        totalLoss = reconLoss + self.beta * klLoss
        return totalLoss


#################################
#              VQ-VAE           #
#################################
class vaeVectorQuantizer(torch.nn.Module):
    def __init__(self, nEmbeddings, dEmbeddings, **kwargs):
        super(vaeVectorQuantizer, self).__init__(**kwargs)
        self.dEmbedding = dEmbeddings
        self.nEmbeddings = nEmbeddings

        # Initialize the embeddings which we will quantize.
        embeddings = torch.FloatTensor(self.dEmbedding, self.nEmbeddings).uniform_(-3 ** 0.5, 3 ** 0.5)
        self.register_parameter('embeddings', torch.nn.Parameter(embeddings))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the input shape of the inputs then
        # flatten inputs keeping `embedding_dim` intact
        input_shape = x.shape
        flattened = torch.flatten(x, 1)

        # Quantization.
        codeIndices = self.codeIndices(flattened)
        encodings = torch.nn.functional.one_hot(codeIndices, self.nEmbeddings).type(self.embeddings.dtype)
        quantized = encodings @ torch.transpose(self.embeddings, 0, 1)
        quantized = torch.reshape(quantized, input_shape)

        return quantized

    def codeIndices(self, y: torch.Tensor) -> torch.Tensor:
        # Calculate L2-norm between inputs and codes
        distances = -2 * y @ self.embeddings
        distances += torch.sum(y * y, dim=1, keepdim=True)
        distances += torch.sum(self.embeddings * self.embeddings, dim=0, keepdim=True)

        # Derive the indices for minimum distances.
        return torch.argmin(distances, axis=1)


class vqVariationalAutoencoder(baseTorchModel):
    def __init__(self, encoder, decoder,
                 nDims, nEmbeddings,
                 beta=0.25, data_var=1.0, **kwargs):
        # initialization
        super(vqVariationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # hidden states
        self.nDims = nDims
        self.nEmbeddings = nEmbeddings

        # parameters
        self.data_var = data_var
        self.beta = beta  # This parameter is best kept between [0.1, 2.0] as per the paper.

        self.vq = vaeVectorQuantizer(nEmbeddings, nDims)

        self.totalLossTracker = baseLossTracker(name="totalLoss")
        self.reconLossTracker = baseLossTracker(name="reconLoss")
        self.codingLossTracker = baseLossTracker(name="codingLoss")
        self.commitLossTracker = baseLossTracker(name="commitmentLoss")
        self.vqLossTracker = baseLossTracker(name="vqLoss")

    @property
    def metrics(self):
        return [self.totalLossTracker,
                self.reconLossTracker,
                self.codingLossTracker,
                self.commitLossTracker,
                self.vqLossTracker]

    def encode(self, x):
        return self.vq(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.encode(x)

    def reconstruct(self, x):
        return self.decoder(self.encode(x))

    def trainStep(self, data, optimizer):
        X = data.to(self.device)
        z_e = self.encoder(X)
        z_q = self.vq(z_e)

        # Calculate vector quantization losses
        codeLoss = torch.mean(torch.sum(torch.flatten(z_q - z_e.detach(), 1) ** 2, dim=-1))  # codebook loss
        commLoss = torch.mean(torch.sum(torch.flatten(z_q.detach() - z_e, 1) ** 2, dim=-1))  # commitment loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # reconstruction
        recon = self.decode(z_q)

        # reconstruction loss
        reconLoss = torch.mean(torch.sum(torch.flatten(X - recon, 1) ** 2, dim=-1))

        # total loss
        totalLoss = reconLoss / self.data_var + codeLoss + self.beta * commLoss

        # Backpropagation
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

        # Loss tracking
        self.totalLossTracker.updateState(totalLoss)
        self.reconLossTracker.updateState(reconLoss)
        self.codingLossTracker.updateState(codeLoss)
        self.commitLossTracker.updateState(commLoss)
        self.vqLossTracker.updateState(codeLoss + self.beta * commLoss)

        # Log results.
        return {"totalLoss": self.totalLossTracker.result(),
                "reconLoss": self.reconLossTracker.result(),
                "codeLoss": self.codingLossTracker.result(),
                "commLoss": self.commitLossTracker.result(),
                "vqLoss": self.vqLossTracker.result()}

    def validStep(self, data):
        X = data.to(self.device)
        z_e = self.encoder(x)
        z_q = self.vq(z_e)

        # Calculate vector quantization losses
        codeLoss = torch.mean(torch.sum(torch.flatten(z_q - z_e.detach(), 1) ** 2, dim=-1))  # codebook loss
        commLoss = torch.mean(torch.sum(torch.flatten(z_q.detach() - z_e, 1) ** 2, dim=-1))  # commitment loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        # reconstruction
        recon = self.decode(z_q)

        # reconstruction loss
        reconLoss = torch.mean(torch.sum(torch.flatten(X - recon, 1) ** 2, dim=-1))

        # total loss
        totalLoss = reconLoss / self.data_var + codeLoss + self.beta * commLoss

        return totalLoss


#################################
#             AVAE              #
#################################
class autoencodingVariationalAutoencoder(baseTorchModel):
    """
    Autoencoding Variational Autoencoder implementation.
    https://arxiv.org/abs/2012.03715
    """
    def __init__(self, encoder: vaeEncoder, decoder: vaeDecoder, rho: float = 0.99, **kwargs):
        # initialization
        super(autoencodingVariationalAutoencoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.register_module('encoder_module', self.encoder.module)
        self.register_module('decoder_module', self.decoder.module)

        assert (rho <= 1.0)
        self.rho = rho
        self.rhosqr = rho * rho

        self.totalLossTracker = baseLossTracker(name="total_loss")
        self.reconLossTracker = baseLossTracker(name="recon_loss")
        self.klLossTracker = baseLossTracker(name="kl_loss")
        self.condLossTracker = baseLossTracker(name="cond_loss")

    @property
    def metrics(self):
        return [self.totalLossTracker,
                self.reconLossTracker,
                self.klLossTracker]

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logVar = torch.split(self.encoder(x), split_size_or_sections=2, dim=1)
        return mean, logVar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zMean, zLogVar = self.encode(inputs)  # get the mean and variance
        z = torch.normal(torch.zeros(zMean.shape), torch.ones(zMean.shape)).to(self.device)  # sample normal RV
        recon = self.decoder(zMean + torch.exp(zLogVar / 2.) * z)  # Reparameterization trick!
        return recon, zMean, zLogVar

    def trainStep(self, data, optimizer):
        # implement the abstract method for a single training update
        X = data.to(self.device)

        # auxiliary variables
        auxX, _, _ = self.forward(X)
        auxX = auxX.detach()
        auxzMean, auxzlogVar = self.encode(auxX)

        recon, zMean, zLogVar = self.forward(X)

        reconLoss = self.decoder.reconstructionLoss(X, recon)
        klLoss = self.encoder.klLoss(zMean, zLogVar)
        condLoss = torch.exp(auxzlogVar) + self.rhosqr * torch.exp(zLogVar)
        condLoss += torch.square(auxzMean - self.rho * zMean)
        condLoss = torch.mean(torch.sum(condLoss, dim=1)) / (2.0 * (1 - self.rhosqr)) - torch.mean(torch.mean(auxzlogVar, dim=1)) / 2.0
        totalLoss = reconLoss + klLoss + condLoss

        # Backpropagation
        totalLoss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.totalLossTracker.updateState(totalLoss)
        self.reconLossTracker.updateState(reconLoss)
        self.klLossTracker.updateState(klLoss)
        self.condLossTracker.updateState(condLoss)

        return {"totalLoss": self.totalLossTracker.result(),
                "reconLoss": self.reconLossTracker.result(),
                "klLoss": self.klLossTracker.result(),
                "condLoss": self.condLossTracker.result()}

    def validStep(self, data):
        # implement the abstract method for a single training update
        X = data.to(self.device)

        # auxiliary variables
        auxX, _, _ = self.forward(X)
        auxX = auxX.detach()
        auxzMean, auxzlogVar = self.encode(auxX)

        recon, zMean, zLogVar = self.forward(X)

        reconLoss = self.decoder.reconstructionLoss(X, recon)
        klLoss = self.encoder.klLoss(zMean, zLogVar)
        condLoss = torch.exp(auxzlogVar) + self.rhosqr * torch.exp(zLogVar)
        condLoss += torch.square(auxzMean - self.rho * zMean)
        condLoss = torch.mean(torch.sum(condLoss, dim=1)) / (2.0 * (1 - self.rhosqr)) - torch.mean(torch.mean(auxzlogVar, dim=1)) / 2.0
        totalLoss = reconLoss + klLoss + condLoss
        return totalLoss


if __name__ == "__main__":
    m = vaeVectorQuantizer(nEmbeddings=10, dEmbeddings=100)
    input = torch.randn(20, 25, 2, 2)
    output = m(input)

