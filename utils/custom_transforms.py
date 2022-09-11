import numpy as np
import utils.h5_utils as h5
from utils.constants import *
from npyx.corr import acg as make_acg


class SwapChannels(object):
    """
    Swaps the indices of even and odd channels, mimicking the biological
    scenario in which the probe was oriented in the same way along the longitudinal axis
    but in the opposite way along the dorsoventral axis.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)
            # Handle the case when we only use a waveform dataset
            if len(data_point.ravel()) == N_CHANNELS * CENTRAL_RANGE:
                wvf = data_point
                acg = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)
            evens = wvf[1::2, :]
            odds = wvf[::2, :]
            new_wvf = np.array([(i, j) for i, j in zip(odds, evens)]).ravel()
            new_data_point = (
                np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")
            )

            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class VerticalReflection(object):
    """
    Reverses the indices of the waveform channels,
    mimicking the  scenario in which the probe was oriented in the same way along
    the dorsoventral axis but in the opposite way along the longitudinal axis.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use a waveform dataset
            if len(data_point.ravel()) == N_CHANNELS * CENTRAL_RANGE:
                wvf = data_point
                acg = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            new_wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)[::-1].ravel().copy()
            new_data_point = (
                np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class GaussianNoise(object):
    """Adds random Gaussian noise to the image."""

    def __init__(self, p=0.3, eps_multiplier=1):
        self.eps = eps_multiplier
        self.p = p

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
                acg = data_point
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            wvf_std = np.std(wvf)
            acg_std = np.std(acg)

            new_wvf = wvf + np.random.normal(0, self.eps * wvf_std, wvf.shape)

            if label == "PkC_ss" or label == 5:
                new_acg = acg
                new_acg = np.clip(new_acg, 0, None)
            else:
                new_acg = acg + np.random.normal(0, self.eps * acg_std, acg.shape)
                new_acg = np.clip(new_acg, 0, None)
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((new_wvf, new_acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class DeleteSpikes(object):
    """Deletes a random portion of the spikes in an ACG"""

    def __init__(self, p=0.3, deletion_prob=0.1):
        self.p = p
        self.deletion_prob = deletion_prob

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]

            new_spikes = spikes[
                np.random.choice(
                    [0, 1],
                    size=(spikes.shape[0]),
                    p=[self.deletion_prob, 1 - self.deletion_prob],
                ).astype(bool)
            ]
            new_acg = make_acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, new_spikes
        else:
            return sample, spikes


class MoveSpikes(object):
    """Randomly moves the spikes in a spike train by a maximum amount"""

    def __init__(self, p=0.3, max_shift=10):
        self.p = p
        self.max_shift = int(np.ceil(max_shift))  # To work with RandAugment behavior

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]

            random_moving = np.random.choice(
                np.arange(-self.max_shift, self.max_shift), size=spikes.shape[0]
            )
            new_spikes = (spikes + random_moving).astype(int)
            new_acg = make_acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
            )
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

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]

            random_addition = np.random.randint(
                low=spikes[0],
                high=spikes[-1],
                size=int(spikes.shape[0] * self.max_addition),
            )
            new_spikes = np.unique(np.concatenate((spikes, random_addition)))
            new_acg = make_acg("hello", 4, 1, 200, train=new_spikes)
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, new_spikes
        else:
            return sample, spikes


class ConstantShift(object):
    """Randomly compresses or expands the signal by a given scalar amount."""

    def __init__(self, p=0.3, scalar=0.1):
        self.scalar = scalar
        self.p = p

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
                acg = data_point
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            acg_amount = self.scalar * np.mean(acg)
            wvf_amount = self.scalar * np.mean(wvf)

            new_acg = (
                (acg_amount + acg)
                if np.random.choice([0, 1]) == 1
                else (acg - acg_amount)
            )
            new_acg = np.clip(new_acg, 0, None)
            new_wvf = (
                (wvf_amount + wvf)
                if np.random.choice([0, 1]) == 1
                else (wvf - wvf_amount)
            )

            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((new_wvf, new_acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class DeleteChannels(object):
    """Randomly delete some channels in the recording.

    Args:
        p (float): Probability of deleting a channel.
        n_channels (int): Number of channels to delete.
    """

    def __init__(self, p=0.3, n_channels=1):
        self.p = p
        self.n_channels = int(np.ceil(n_channels))  # To work with RandAugment behavior

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use a waveform dataset
            if len(data_point.ravel()) == N_CHANNELS * CENTRAL_RANGE:
                wvf = data_point
                acg = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)
            deleted_channels = np.random.choice(
                np.arange(wvf.shape[0]),
                size=self.n_channels,
                replace=False,
            )
            noise = np.random.rand(wvf.shape[1]) * np.std(wvf[0, :])
            new_wvf = wvf.copy()
            new_wvf[deleted_channels, :] = noise
            new_wvf = new_wvf.ravel()
            new_data_point = (
                np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")
            )

            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class NewWindowACG(object):
    """Recomputes the given acg with a different bin_size and window_size."""

    def __init__(self, p=0.3, magnitude_change=3):
        self.p = p
        self.magnitude_change = magnitude_change

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use an ACG Dataset
            if len(data_point.ravel()) == ACG_LEN:
                wvf = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]

            new_acg = make_acg(
                "hello",
                4,
                (0.5 * self.magnitude_change),
                (100 * self.magnitude_change),
                train=spikes,
            )
            new_acg = h5.resample_acg(new_acg[int(len(new_acg) / 2) :]) / ACG_SCALING
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is wrong in class {self.__name__}"
            new_data_point = (
                np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
            )
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class PermuteChannels(object):
    """Randomly permutes some channels in the recording.

    Args:
        p (float): Probability of applying the permutation.
        n_channels (int): Number of channels to permute.
    """

    def __init__(self, p=0.3, n_channels=1):
        self.p = p
        self.n_channels = int(np.ceil(n_channels))  # To work with RandAugment behavior
        assert self.n_channels <= N_CHANNELS // 2, "Too many channels to permute"

    def __call__(self, sample, spikes):
        if self.p > np.random.rand():
            data_point, label = sample
            data_point = np.squeeze(data_point)

            # Handle the case when we only use a waveform dataset
            if len(data_point.ravel()) == N_CHANNELS * CENTRAL_RANGE:
                wvf = data_point
                acg = np.array([])
            else:
                wvf = data_point[: N_CHANNELS * CENTRAL_RANGE]
                acg = data_point[N_CHANNELS * CENTRAL_RANGE :]

            wvf = wvf.reshape(N_CHANNELS, CENTRAL_RANGE)
            permuted_channels = np.random.choice(
                np.arange(wvf.shape[0]), size=self.n_channels * 2, replace=False
            )

            new_wvf = wvf.copy()
            new_wvf[permuted_channels[: self.n_channels]] = wvf[
                permuted_channels[self.n_channels :]
            ]
            new_wvf[permuted_channels[self.n_channels :]] = wvf[
                permuted_channels[: self.n_channels]
            ]

            new_wvf = new_wvf.ravel()
            new_data_point = (
                np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")
            )

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
