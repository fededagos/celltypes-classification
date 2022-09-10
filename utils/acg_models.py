import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import *
import npyx
import utils.h5_utils as h5


class ForwardFullEncoder(nn.Module):
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
        self.fc1 = nn.LazyLinear(250)
        self.fc2 = nn.LazyLinear(125)
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

        return dist.Normal(mu, torch.clip(std, 1e-5, 1e5))


class ForwardFullDecoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Encoder `nn.Module`.

        This will operate on inputs of shape (batch_size, 1, 28, 28).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.dec_fc1 = nn.Linear(D_latent, 125)
        self.dec_fc2 = nn.LazyLinear(250)
        self.dec_dropout = nn.Dropout(p=0.1)
        self.dec_fc3 = nn.LazyLinear((N_CHANNELS * CENTRAL_RANGE + ACG_LEN) * 2)
        self.flatten = nn.Flatten()

    def forward(self, Z):
        """Return a distribution p(x | z)

        INPUT:
        X    : torch.FloatTensor, real-valued, shape = (batch_size, D_latent)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, 1, height, width)
        """

        # Make sure that the returned value has the right shape! e.g.:
        # return dist.Bernoulli(X_hat.reshape(-1, 1, 28, 28))

        # YOUR CODE HERE
        h = F.relu(self.dec_fc1(Z))
        h = F.relu(self.dec_fc2(h))
        h = self.dec_dropout(h)
        h = self.dec_fc3(h)
        X_hat = h[:, : N_CHANNELS * CENTRAL_RANGE + ACG_LEN]
        log_sig = h[:, N_CHANNELS * CENTRAL_RANGE + ACG_LEN :]
        sig = log_sig.exp()

        return dist.Normal(
            X_hat.reshape(-1, 1, N_CHANNELS * CENTRAL_RANGE + ACG_LEN),
            torch.clip(sig, 1e-5, 1e5).reshape(
                -1, 1, N_CHANNELS * CENTRAL_RANGE + ACG_LEN
            ),
        )


class ForwardACGEncoder(nn.Module):
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
        self.fc1 = nn.LazyLinear(250)
        self.fc2 = nn.LazyLinear(125)
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

        return dist.Normal(mu, torch.clip(std, 1e-5, 1e5))


class ForwardACGDecoder(nn.Module):
    def __init__(self, D_latent):
        """
        Initialize the Encoder `nn.Module`.

        This will operate on inputs of shape (batch_size, 1, 28, 28).

        INPUTS:
        D_latent: size of latent space (integer)

        """
        super().__init__()
        self.D_latent = D_latent

        self.dec_fc1 = nn.Linear(D_latent, 125)
        self.dec_fc2 = nn.LazyLinear(250)
        self.dec_dropout = nn.Dropout(p=0.1)
        self.dec_fc3 = nn.LazyLinear(ACG_LEN * 2)
        self.flatten = nn.Flatten()

    def forward(self, Z):
        """Return a distribution p(x | z)

        INPUT:
        X    : torch.FloatTensor, real-valued, shape = (batch_size, D_latent)

        OUTPUT: a `torch.Distribution` instance, defined on values of shape = (batch_size, 1, height, width)
        """

        # Make sure that the returned value has the right shape! e.g.:
        # return dist.Bernoulli(X_hat.reshape(-1, 1, 28, 28))

        # YOUR CODE HERE
        h = F.relu(self.dec_fc1(Z))
        h = F.relu(self.dec_fc2(h))
        h = self.dec_dropout(h)
        h = self.dec_fc3(h)
        X_hat = h[:, :ACG_LEN]
        log_sig = h[:, ACG_LEN:]
        sig = log_sig.exp()

        return dist.Normal(
            X_hat.reshape(-1, 1, ACG_LEN),
            torch.clip(sig, 1e-5, 1e5).reshape(-1, 1, ACG_LEN),
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
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
            acg = data_point[N_CHANNELS * CENTRAL_RANGE :]
            wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)
            evens = wvf[1::2, :]
            odds = wvf[::2, :]
            new_wvf = np.array([(i, j) for i, j in zip(odds, evens)]).ravel()
            new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1)

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
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
            acg = data_point[N_CHANNELS * CENTRAL_RANGE :]
            new_wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)[::-1].ravel().copy()

            new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1)
            transformed_sample = (new_data_point, label)
            return transformed_sample
        else:
            return sample


class DeleteSpikes(object):
    """Deletes a random portion of the spikes in an ACG"""

    def __init__(self, p=0.3, deletion_prob=0.1):
        self.p = p
        self.deletion_prob = deletion_prob

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            new_spikes = spikes[
                np.random.choice(
                    [0, 1],
                    size=(spikes.shape[0]),
                    p=[self.deletion_prob, 1 - self.deletion_prob],
                ).astype(bool)
            ]
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = new_acg.reshape(1, -1).astype("float32")
            transformed_sample = (new_data_point, label)
            return transformed_sample, new_spikes
        else:
            return sample, spikes


class MoveSpikes(object):
    """Randomly moves the spikes in a spike train by a maximum amount"""

    def __init__(self, p=0.3, max_shift=10):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            random_moving = np.random.choice(
                np.arange(-self.max_shift, self.max_shift), size=spikes.shape[0]
            )
            new_spikes = (spikes + random_moving).astype(int)
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = new_acg.reshape(1, -1).astype("float32")
            transformed_sample = (new_data_point, label)
            return transformed_sample, new_spikes
        else:
            return sample, spikes


class AddSpikes(object):
    """Adds a random amount of spikes (in percentage) to the spike list and recomputes the ACG"""

    def __init__(self, p=0.3, max_addition=0.1):
        self.p = p
        self.max_addition = max_addition

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            random_addition = np.random.randint(
                low=spikes[0],
                high=spikes[-1],
                size=int(spikes.shape[0] * self.max_addition),
            )
            new_spikes = np.unique(np.concatenate((spikes, random_addition)))
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = new_acg.reshape(1, -1).astype("float32")
            transformed_sample = (new_data_point, label)
            return transformed_sample, new_spikes
        else:
            return sample, spikes


class ConstantShift(object):
    """Randomly compresses or expands the signal by a given scalar amount."""

    def __init__(self, scalar=0.1, p=0.3):
        self.scalar = scalar
        self.p = p

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            acg = data_point
            acg_amount = self.scalar * np.mean(acg)

            new_acg = (
                (acg_amount + acg)
                if np.random.choice([0, 1]) == 1
                else (acg - acg_amount)
            )
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = new_acg.reshape(1, -1).astype("float32")
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class CustomCompose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, spikes):
        for t in self.transforms:
            sample, spikes = t(sample, spikes)
        return sample, spikes

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
