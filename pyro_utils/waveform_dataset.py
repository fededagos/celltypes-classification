from argon2 import Type
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F

import numpy as np
from functools import reduce
from pyro_utils.custom_mlp import MLP, Exp
from pyro_utils.mnist_cached import MNISTCached, mkdir_p, setup_data_loaders
from pyro_utils.vae_plots import mnist_test_tsne_ssvae, plot_conditional_samples_ssvae
from visdom import Visdom
from sklearn.model_selection import train_test_split

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)
from pyro.optim import Adam
from pyro.contrib.examples.util import MNIST, get_data_directory

### Constants

# Notes: train and test size refer to (labelled + unlabelled) size.
# The amount of labels used is going to be determined by the sup_num parameter in the
# Dataset class.

CENTRAL_RANGE = 60

N_CHANNELS = 10

TRAIN_SIZE = 1000

TEST_SIZE = 0

VALIDATION_SIZE = 100

BATCH_SIZE = 50

ACG_LEN = 100

N_CLASSES = 6

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###

# transformations for MNIST data
def flatten_x(x, use_cuda):

    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    x_1d_size = reduce(lambda a, b: a * b, x.size()[1:])
    x = x.view(-1, x_1d_size)

    # send the data to GPU(s)
    if use_cuda:
        x = x.cuda()

    return x


def one_hot(y, classes=6):
    """Produces a one_hot encoding of the given vector of labels.

    Args:
    y (ndarray): vector of labels to encode
    classes (int): number of classes to encode represented in y. Default is 6 for our case.

    Returns:
    Y_one_hot (ndarray): matrix of shape (len(y), classes) of one hot encoded y
    """
    try:
        Y_one_hot = np.zeros((len(y), classes))
        Y_one_hot[range(y.shape[0]), (y.astype(int))] = 1
    except TypeError:
        Y_one_hot = np.zeros((classes))
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

    X, X_valid, y, y_valid = train_test_split(X, y, test_size=0.1, stratify=y)

    assert sup_num % N_CLASSES == 0, "unable to have equal number of images per class"

    # number of supervised examples per class
    sup_per_class = int(sup_num / N_CLASSES)

    idxs_sup, idxs_unsup = get_ss_indices_per_class(one_hot(y), sup_per_class)
    X_sup = X[idxs_sup]
    y_sup = one_hot(y[idxs_sup])
    #! Mind here that we are putting the extra labels into the validation set not to waste them
    X_valid = np.concatenate((X[idxs_unsup], X_valid), axis=0)
    y_valid = one_hot(np.concatenate((y[idxs_unsup], y_valid), axis=0))

    return X_sup, y_sup, unlab_X, one_hot(unlab_y), X_valid, y_valid


def setup_data_loaders(
    dataset, use_cuda, batch_size, data_points, targets, sup_num=None, **kwargs
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
            use_cuda=use_cuda,
        )
        loaders[mode] = data.DataLoader(
            cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs
        )

    return loaders


class WaveformDataset(data.Dataset):
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

        assert mode in [
            "sup",
            "unsup",
            "valid",
        ], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if WaveformDataset.train_data_sup is None:
                if sup_num is None:
                    assert mode == "unsup"
                    (
                        WaveformDataset.train_data_unsup,
                        WaveformDataset.train_labels_unsup,
                    ) = (
                        self.data,
                        self.targets,
                    )
                else:
                    (
                        WaveformDataset.train_data_sup,
                        WaveformDataset.train_labels_sup,
                        WaveformDataset.train_data_unsup,
                        WaveformDataset.train_labels_unsup,
                        WaveformDataset.data_valid,
                        WaveformDataset.labels_valid,
                    ) = split_sup_unsup_valid(self.data, self.targets, sup_num)

            if mode == "sup":
                self.data, self.targets = (
                    WaveformDataset.train_data_sup,
                    WaveformDataset.train_labels_sup,
                )
            elif mode == "unsup":
                self.data = WaveformDataset.train_data_unsup

                # making sure that the unsupervised labels are not available to inference
                self.targets = (
                    torch.Tensor(WaveformDataset.train_labels_unsup.shape[0]).view(
                        -1, 1
                    )
                ) * np.nan
            else:
                self.data, self.targets = (
                    WaveformDataset.data_valid,
                    WaveformDataset.labels_valid,
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data_point = self.data[index].astype("float32")  # .reshape(1, -1)
        label = self.targets[index]

        sample = (data_point, label)
        if self.transform is not None:
            sample = self.transform((data_point, label))

        return sample


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
                np.array([(i, j) for i, j in zip(odds, evens)]).ravel()
                # .reshape(1, -1)
                # .reshape(1, N_CHANNELS, CENTRAL_RANGE)
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
                .ravel()
                # .reshape(1, -1)
                # .reshape(1, N_CHANNELS, CENTRAL_RANGE)
            )
            transformed_sample = (data_point, label)
            return transformed_sample
        else:
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
