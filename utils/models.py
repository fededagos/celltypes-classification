import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import *


class ConvEncoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Encoder `nn.Module`.

        This will operate on inputs of shape (batch_size, 1, 28, 28).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.norm = nn.BatchNorm2d(1)
        self.enc_conv_1 = nn.Conv2d(1, 3, (2, 4))
        self.enc_conv_2 = nn.Conv2d(3, 6, (2, 4))
        self.enc_dropout = nn.Dropout(p=0.1)
        self.enc_flatten = nn.Flatten()
        self.fc_params = nn.Linear(2592, D_latent + D_latent)

    def forward(self, X):
        """Return a distribution q(z | x).

        INPUT:
        X    : torch.FloatTensor containing zeros and ones; shape = (batch_size, 1, 28, 28)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, D_latent)
        """
        D_latent = self.D_latent

        s = self.norm(X)
        s = F.relu(self.enc_conv_1(s))
        s = F.relu(self.enc_conv_2(s))
        s = self.enc_dropout(s)
        flat_s = self.enc_flatten(s)
        params = self.fc_params(flat_s)
        mu, std = (
            params[:, :-D_latent],
            params[:, -D_latent:].exp(),
        )  # exp to enforce positive

        assert mu.shape == (
            X.shape[0],
            self.D_latent,
        )  # Keeping assertions as sanity checks
        assert mu.shape == std.shape

        return dist.Normal(mu, std)


class ConvDecoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Decoder `nn.Module`.

        This will operate on inputs of shape (batch_size, D_latent).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.dec_fc = nn.Linear(D_latent, 2592)
        self.dec_unflatten = nn.Unflatten(-1, (6, 8, 54))
        self.dec_dropout = nn.Dropout(p=0.1)
        self.dec_deconv_1 = nn.ConvTranspose2d(6, 3, kernel_size=(2, 4))
        self.dec_deconv_2 = nn.ConvTranspose2d(3, 1, kernel_size=(2, 4))

        self.log_sig = nn.Parameter(
            torch.zeros((BATCH_SIZE, 1, N_CHANNELS, CENTRAL_RANGE))
        )  # to use in reconstruction

    def forward(self, Z):
        """Return a distribution p(x | z)

        INPUT:
        X    : torch.FloatTensor, real-valued, shape = (batch_size, D_latent)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, 1, 28, 28)
        """


        s = F.relu(self.dec_fc(Z))
        s = self.dec_unflatten(s)
        s = self.dec_dropout(s)
        s = F.relu(self.dec_deconv_1(s))
        s = self.dec_deconv_2(s)
        X_hat = s

        return dist.Laplace(
            X_hat.reshape(-1, 1, N_CHANNELS, CENTRAL_RANGE), self.log_sig.exp()
        )


class ForwardEncoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Encoder `nn.Module`.

        This will operate on inputs of shape (batch_size, 1, 28, 28).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.flat_s = nn.Flatten()
        self.fc1 = nn.LazyLinear(200)
        self.fc2 = nn.LazyLinear(100)
        self.fc3 = nn.LazyLinear(D_latent + D_latent)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        """Return a distribution q(z | x).

        INPUT:
        X    : torch.FloatTensor containing zeros and ones; shape = (batch_size, 1, heigth, width)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, D_latent)
        """
        D_latent = self.D_latent

        h1 = self.flat_s(X)
        h2 = F.relu(self.fc1(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc2(h2))
        params = self.fc3(h3)
        mu, std = (
            params[:, :-D_latent],
            params[:, -D_latent:].exp(),
        )  # exp to enforce positive

        assert mu.shape == (
            X.shape[0],
            self.D_latent,
        )  # Keeping assertions as sanity checks
        assert mu.shape == std.shape

        return dist.Normal(mu, std)


class ForwardDecoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Encoder `nn.Module`.

        This will operate on inputs of shape (batch_size, 1, 28, 28).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.dec_fc1 = nn.LazyLinear(100)
        self.dec_fc2 = nn.LazyLinear(200)
        self.dec_dropout = nn.Dropout(p=0.1)
        self.dec_fc3 = nn.LazyLinear(N_CHANNELS * CENTRAL_RANGE * 2)
        self.flatten = nn.Flatten()

    def forward(self, Z):
        """Return a distribution p(x | z)

        INPUT:
        X    : torch.FloatTensor, real-valued, shape = (batch_size, D_latent)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, 1, height, width)
        """

        h = F.relu(self.dec_fc1(Z))
        h = F.relu(self.dec_fc2(h))
        h = self.dec_dropout(h)
        h = self.dec_fc3(h)
        X_hat = h[:, : N_CHANNELS * CENTRAL_RANGE]
        log_sig = h[:, N_CHANNELS * CENTRAL_RANGE :]

        return dist.Normal(
            X_hat.reshape(-1, 1, N_CHANNELS, CENTRAL_RANGE),
            log_sig.reshape(-1, 1, N_CHANNELS, CENTRAL_RANGE).exp(),
        )


class SwapChannels(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample):
        data_point, label = sample

        if self.p > np.random.rand():
            data_point = data_point.reshape(N_CHANNELS, CENTRAL_RANGE)
            evens = data_point[1::2, :]
            odds = data_point[::2, :]
            new_data_point = (
                np.array([(i, j) for i, j in zip(odds, evens)])
                .ravel()
                .reshape(1, N_CHANNELS, CENTRAL_RANGE)
            )

            transformed_sample = (new_data_point, label)
            return transformed_sample
        else:
            return sample


class VerticalReflection(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample):
        data_point, label = sample
        if self.p > np.random.rand():
            data_point = (
                data_point.reshape(N_CHANNELS, CENTRAL_RANGE)[::-1]
                .copy()
                .reshape(1, N_CHANNELS, CENTRAL_RANGE)
            )
            transformed_sample = (data_point, label)
            return transformed_sample
        else:
            return sample


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 117)
        self.dropout1 = nn.Dropout(0.223957828021243)
        self.hidden_fc = nn.Linear(117, 148)
        self.dropout2 = nn.Dropout(0.401563000009004)
        self.output_fc = nn.Linear(148, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.dropout1(self.input_fc(x)))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.dropout2(self.hidden_fc(h_1)))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred
