# Custom wrappers for FixMatch mad to work with the cerebellum dataset.
# Some functions are borrowed from https://github.com/kekmodel/FixMatch-pytorch
import logging
import math
import torch
import numpy as np
import utils.custom_transforms as transforms
import torch.utils.data as data
import random
import utils.h5_utils as h5
from utils.constants import DATA_PATH
import copy


logger = logging.getLogger(__name__)


def leave_labels_out(input_dataset: h5.NeuronsDataset, n_lab_per_class=4):
    labelled_mask = np.array(input_dataset.targets) != -1
    indexes = np.arange(len(input_dataset.targets))
    labelled_indexes = indexes[labelled_mask]
    labels = input_dataset.targets[labelled_indexes]

    choices = []
    for label in np.unique(labels):
        label_indices = labelled_indexes[labels == label]
        choose = np.random.choice(label_indices, size=n_lab_per_class, replace=False)
        choices.append(choose)
    choices = np.concatenate(choices)

    bool_masks = []
    for element in choices:
        bool_mask = element == labelled_indexes
        bool_masks.append(bool_mask[:, None])

    final_mask = np.stack(bool_masks, axis=2).sum(2).ravel().astype(bool)
    mask_out = labelled_indexes[~final_mask]

    train_dataset = copy.copy(input_dataset)
    test_lab_only = copy.copy(input_dataset)

    train_dataset.wf = np.delete(train_dataset.wf, mask_out, axis=0)
    train_dataset.acg = np.delete(train_dataset.acg, mask_out, axis=0)
    train_dataset.spikes_list = np.delete(
        np.array(train_dataset.spikes_list, dtype="object"), mask_out, axis=0
    ).tolist()
    train_dataset.targets = np.delete(train_dataset.targets, mask_out, axis=0)
    train_dataset.full_dataset = np.delete(train_dataset.full_dataset, mask_out, axis=0)

    test_lab_only.wf = test_lab_only.wf[mask_out]
    test_lab_only.acg = test_lab_only.acg[mask_out]
    test_lab_only.spikes_list = np.array(test_lab_only.spikes_list, dtype="object")[
        mask_out
    ].tolist()
    test_lab_only.targets = test_lab_only.targets[mask_out]
    test_lab_only.full_dataset = test_lab_only.full_dataset[mask_out]

    return train_dataset, test_lab_only


def get_cerebellum_dataset(args):
    transform_labeled = transforms.CustomCompose(
        [
            transforms.SwapChannels(p=0.5),
            transforms.VerticalReflection(p=0.3),
        ]
    )
    transform_val = transforms.CustomCompose(
        [
            transforms.SwapChannels(p=0.5),
            transforms.VerticalReflection(p=0.3),
        ]
    )

    base_dataset = h5.NeuronsDataset(DATA_PATH, quality_check=True, normalise=False)
    # labels_only_dataset = copy.copy(base_dataset)
    base_dataset.min_max_scale()
    base_dataset.make_full_dataset(args.wvf_only, args.acg_only)

    # labels_only_dataset.min_max_scale()
    # labels_only_dataset.make_labels_only()
    # labels_only_dataset.make_full_dataset(args.wvf_only, args.acg_only)

    train_dataset, labels_only_dataset = leave_labels_out(
        base_dataset, n_lab_per_class=args.num_labeled // args.num_classes
    )

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, train_dataset.targets)

    train_labeled_dataset = CerebellumDatasetSSL(
        train_dataset.full_dataset,
        train_dataset.targets,
        train_dataset.spikes_list,
        train_labeled_idxs,
        transform=transform_labeled,
    )

    train_unlabeled_dataset = CerebellumDatasetSSL(
        train_dataset.full_dataset,
        train_dataset.targets,
        train_dataset.spikes_list,
        train_unlabeled_idxs,
        transform=TransformFixMatch(),
    )

    test_dataset = CerebellumDatasetSSL(
        labels_only_dataset.full_dataset,
        labels_only_dataset.targets,
        labels_only_dataset.spikes_list,
        indexs=None,
        transform=transform_val,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.CustomCompose(
            [
                transforms.SwapChannels(p=0.5),
                transforms.VerticalReflection(p=0.3),
            ]
        )
        self.strong = RandAugmentMC(n=4, m=10)

    def __call__(self, sample, spikes):
        weak_sample, _ = self.weak(sample, spikes)
        weak_data_point, _ = weak_sample
        strong_data_point, _ = self.strong(
            sample, spikes
        )  # Because we return samples in our transform but we only need the data!
        return (weak_data_point, strong_data_point), _


def squeeze_output(func):
    """Decorator to squeeze the output of a class method."""

    def func_wrapper(*args, **kwargs):
        sample, label = func(*args, **kwargs)
        new_sample = sample.squeeze()
        return (new_sample, label)

    return func_wrapper


class CerebellumDatasetSSL(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, CENTRAL_RANGE)"""

    def __init__(self, data, targets, raw_spikes, indexs=None, transform=None):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            targets (string): Array of targets for the provided data
            raw_spikes (ndarray): Array of raw spikes for the provided data
        """
        self.data = data
        self.targets = targets
        self.transform = transform
        self.spikes = raw_spikes
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.spikes = [
                self.spikes[i] for i in indexs
            ]  # It is a list so cannot work with numpy indexing here

    def __len__(self):
        return len(self.data)

    # @squeeze_output
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = self.data[idx, :].astype("float32").reshape(1, -1)
        target = self.targets[idx].astype("int")
        spikes = self.spikes[idx].astype("int")
        sample = (data_point, target)

        if self.transform is not None:
            sample, spikes = self.transform(sample, spikes)
        (
            data_point,
            target,
        ) = sample  # This unpacking is to make it work as expected with the randaugment!
        return data_point, target


def fixmatch_augment_pool():
    # All custom augmentations for our dataset
    # The triplets stand for p of applying the transform and max magnitude for that transform
    augs = [
        (
            transforms.AddSpikes,
            1,
            0.6,
        ),
        (transforms.DeleteSpikes, 1, 0.6),
        (transforms.ConstantShift, 1, 0.4),
        (transforms.GaussianNoise, 1, 2),
        (transforms.MoveSpikes, 1, 30),
        (transforms.DeleteChannels, 1, 5),
        (transforms.NewWindowACG, 1, 3),
        (
            transforms.PermuteChannels,
            1,
            5,
        ),
    ]
    # TODO We got to 8 custom transforms so far. There were 13 in the original paper
    return augs


def waveform_augment_pool():
    augs = [
        (transforms.DeleteChannels, 1, 5),
        (
            transforms.PermuteChannels,
            1,
            5,
        ),
    ]
    return augs


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, sample, spikes):

        # First we apply the soft transforms
        sample, spikes = transforms.SwapChannels(p=0.6)(sample, spikes)
        sample, spikes = transforms.VerticalReflection(p=0.6)(sample, spikes)

        # Then choose transforms to apply from the augment_pool
        ops = random.choices(self.augment_pool, k=self.n)
        for op, p, max_magn in ops:
            v = np.random.randint(1, self.m)
            v = v / 10  #! To correct for behaviour of our custom transforms
            if random.random() < 0.5:
                augmentation = op(p, max_magn * v)
                sample, spikes = augmentation(sample, spikes)
        data_point, label = sample
        return data_point, spikes


class RandTrans(object):
    def __init__(self, n, m, acg=True):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool() if acg else waveform_augment_pool()

    def __call__(self, sample, spikes):

        # First we apply the soft transforms
        sample, spikes = transforms.SwapChannels(p=0.6)(sample, spikes)
        sample, spikes = transforms.VerticalReflection(p=0.6)(sample, spikes)

        # Then choose transforms to apply from the augment_pool
        ops = random.choices(self.augment_pool, k=self.n)
        for op, p, max_magn in ops:
            v = np.random.randint(1, self.m)
            v = v / 10  #! To correct for behaviour of our custom transforms
            if random.random() < 0.5:
                augmentation = op(p, max_magn * v)
                sample, spikes = augmentation(sample, spikes)
        return sample, spikes
