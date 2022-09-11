import npyx
import torch
import numpy as np
import torch.nn as nn
import utils.h5_utils as h5
from functools import reduce
from utils.constants import *
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def squeeze_output(func):
    """Decorator to squeeze the output of a class method."""

    def func_wrapper(*args, **kwargs):
        sample, label = func(*args, **kwargs)
        new_sample = sample.squeeze()
        return (new_sample, label)

    return func_wrapper


class CerebellumDataset(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, CENTRAL_RANGE)"""

    def __init__(self, data, labels, raw_spikes, transform=None):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            labels (string): Array of labels for the provided data
            raw_spikes (ndarray): Array of raw spikes for the provided data
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.spikes = raw_spikes

    def __len__(self):
        return len(self.data)

    @squeeze_output
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = self.data[idx, :].astype("float32").reshape(1, -1)
        label = self.labels[idx].astype("int")
        spikes = self.spikes[idx].astype("int")
        sample = (data_point, label)

        if self.transform:
            sample, spikes = self.transform(sample, spikes)
        return sample


def flatten_x(x, use_cuda):

    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    x_1d_size = reduce(lambda a, b: a * b, x.size()[1:])
    x = x.view(-1, x_1d_size)

    # send the data to GPU(s)
    if use_cuda:
        x = x.cuda()

    return x


def one_hot(y):
    """Produces a one_hot encoding of the given vector of labels.

    Args:
    y (ndarray): vector of labels to encode
    classes (int): number of classes to encode represented in y. Default is 6 for our case.

    Returns:
    Y_one_hot (ndarray): matrix of shape (len(y), classes) of one hot encoded y
    """
    try:
        Y_one_hot = np.zeros((len(y), N_CLASSES))
        Y_one_hot[range(y.shape[0]), (y.astype(int))] = 1
    except TypeError:
        Y_one_hot = np.zeros((N_CLASSES))
        Y_one_hot[int(y)] = 1
    except AttributeError:
        return y

    return torch.Tensor(Y_one_hot).to(torch.long)


def get_ss_indices_per_class(y, sup_per_class):
    #! Here is really where we should take into account the unlabelled data
    #! and split so that its indices are all assigned into the unsup category
    #! Once again sklearn probably has already utilities for this!
    # number of indices to consider
    n_idxs = y.shape[0]

    # calculate the indices per class
    idxs_per_class = {j: [] for j in range(N_CLASSES)}

    # for each index identify the class and add the index to the right class
    for i in range(n_idxs):
        curr_y = y[i]
        for j in range(N_CLASSES):
            if curr_y[j] == 1:
                idxs_per_class[j].append(i)
                break

    idxs_sup = []
    idxs_unsup = []
    for j in range(N_CLASSES):
        np.random.shuffle(idxs_per_class[j])
        idxs_sup.extend(idxs_per_class[j][:sup_per_class])
        idxs_unsup.extend(idxs_per_class[j][sup_per_class : len(idxs_per_class[j])])

    return idxs_sup, idxs_unsup


def split_sup_unsup_valid(X, y, sup_num, validation_num=VALIDATION_SIZE):
    """
    helper function for splitting the data into supervised, un-supervised and validation parts
    :param X: images
    :param y: labels (digits)
    :param sup_num: what number of examples is supervised
    :param validation_num: what number of last examples to use for validation
    :return: splits of data by sup_num number of supervised examples
    """

    # validation set needs to be on labelled data only, so we first split by this
    unlabelled_mask = y == -1
    unlab_y = y[unlabelled_mask]
    unlab_X = X[unlabelled_mask]
    y = y[~unlabelled_mask]
    X = X[~unlabelled_mask]

    X, X_valid, y, y_valid = train_test_split(
        X, y, test_size=validation_num, stratify=y
    )

    assert (
        sup_num % len(np.unique(y)) == 0
    ), "unable to have equal number of images per class"

    # number of supervised examples per class
    sup_per_class = int(sup_num / len(np.unique(y)))

    idxs_sup, idxs_unsup = get_ss_indices_per_class(one_hot(y), sup_per_class)
    X_sup = X[idxs_sup]
    y_sup = one_hot(y[idxs_sup])
    #! Mind here that we are putting the extra labels into the validation set not to waste them
    X_valid = np.concatenate((X[idxs_unsup], X_valid), axis=0)
    y_valid = one_hot(np.concatenate((y[idxs_unsup], y_valid), axis=0))

    return X_sup, y_sup, unlab_X, unlab_y, X_valid, y_valid


def setup_data_loaders(
    dataset, use_cuda, batch_size, data_points, targets, spikes, sup_num=None, **kwargs
):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param dataset: the data to use
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_num: number of supervised data examples
    :param download: download the dataset (if it doesn't exist already)
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """
    # instantiate the dataset as training/testing sets
    if "num_workers" not in kwargs:
        kwargs = {"num_workers": 0, "pin_memory": False}

    cached_data = {}
    loaders = {}
    for mode in ["unsup", "sup", "valid"]:
        if sup_num is None and mode == "sup":
            # in this special case, we do not want "sup"
            return loaders["unsup"], loaders["valid"]
        cached_data[mode] = dataset(
            mode=mode,
            sup_num=sup_num,
            data=data_points,
            targets=targets,
            spikes=spikes,
            use_cuda=use_cuda,
        )
        loaders[mode] = data.DataLoader(
            cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs
        )

    return loaders


class PyroCerebellumDataset(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, CENTRAL_RANGE)"""

    # static class variables for caching training data
    #! Values need changing
    train_data_size = TRAIN_SIZE
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    validation_size = VALIDATION_SIZE
    data_valid, labels_valid = None, None

    def __init__(
        self,
        data,
        targets,
        spikes,
        mode,
        sup_num,
        use_cuda=False,
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            data (ndarray): Array of data points
            targets (list): Array of targets for the provided data
        """
        self.use_cuda = use_cuda
        self.data = data
        self.targets = targets
        self.mode = mode
        self.train = mode in ["sup", "unsup", "valid"]
        self.transform = transform
        self.target_tansform = target_transform
        self.sup_num = sup_num
        self.spikes = spikes

        if self.use_cuda:
            self.data = self.data.cuda()
            self.targets = self.targets.cuda()

        assert mode in [
            "sup",
            "unsup",
            "valid",
        ], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if PyroCerebellumDataset.train_data_sup is None:
                if sup_num is None:
                    assert mode == "unsup"
                    (
                        PyroCerebellumDataset.train_data_unsup,
                        PyroCerebellumDataset.train_labels_unsup,
                    ) = (
                        self.data,
                        self.targets,
                    )
                else:
                    (
                        PyroCerebellumDataset.train_data_sup,
                        PyroCerebellumDataset.train_labels_sup,
                        PyroCerebellumDataset.train_data_unsup,
                        PyroCerebellumDataset.train_labels_unsup,
                        PyroCerebellumDataset.data_valid,
                        PyroCerebellumDataset.labels_valid,
                    ) = split_sup_unsup_valid(self.data, self.targets, sup_num)

            if mode == "sup":
                self.data, self.targets = (
                    PyroCerebellumDataset.train_data_sup,
                    PyroCerebellumDataset.train_labels_sup,
                )
            elif mode == "unsup":
                self.data = PyroCerebellumDataset.train_data_unsup

                # making sure that the unsupervised labels are not available to inference
                self.targets = (
                    torch.Tensor(
                        PyroCerebellumDataset.train_labels_unsup.shape[0]
                    ).view(-1, 1)
                ) * np.nan
            else:
                self.data, self.targets = (
                    PyroCerebellumDataset.data_valid,
                    PyroCerebellumDataset.labels_valid,
                )

    def __len__(self):
        return len(self.data)

    @squeeze_output
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data_point = self.data[index].astype("float32")  # .reshape(1, -1)
        label = self.targets[index]
        spikes = self.spikes[index].astype("int")

        sample = (data_point, label)
        if self.transform is not None:
            sample, spikes = self.transform(sample, spikes)

        return sample


class OneHot(object):
    """One hot encoding transform for dataset targets."""

    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda

    def __call__(self, target):
        yp = torch.zeros(target.size(0), N_CLASSES)

        # send the data to GPU(s)
        if self.use_cuda:
            yp = yp.cuda()
            target = target.cuda()
        # transform the label y (integer between 0 and 9) to a one-hot
        yp = yp.scatter_(1, target.view(-1, 1), 1.0)
        return yp


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
            new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1)

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
            new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1)
            transformed_sample = (new_data_point, label)
            return transformed_sample, spikes
        else:
            return sample, spikes


class GaussianNoise(object):
    """Adds random Gaussian noise to the image."""

    def __init__(self, eps_multiplier=1, p=0.3):
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
            else:
                new_acg = acg + np.random.normal(0, self.eps * acg_std, acg.shape)
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
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
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
        self.max_shift = max_shift

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
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
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
            new_acg = npyx.corr.acg("hello", 4, 1, 200, train=new_spikes)
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

    def __init__(self, scalar=0.1, p=0.3):
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
