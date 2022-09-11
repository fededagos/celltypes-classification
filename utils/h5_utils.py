import sys, os
import json
from pathlib import Path
import h5py
import npyx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.constants import *
from tqdm.auto import tqdm
import random
import pickle
import copy

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def save(file_name, obj):
    with open(file_name, "wb") as fobj:
        pickle.dump(obj, fobj)


def load(file_name):
    with open(file_name, "rb") as fobj:
        return pickle.load(fobj)


def get_neuron_attr(hdf5_file_path, id=None, file=None):
    """
    Prompts the user to select a given neuron's file to load.
    Otherwise, can specify which neuron's id and which file we want to load directly
    """
    neuron_ids = []
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        for name in hdf5_file:
            if "neuron" in name:
                pi = name.split("_")[0]
                neuron_id = name.split("_")[-1]
                neuron_ids.append(neuron_id)
            else:
                continue
        if id is None:
            neuron_ids = [int(neuron_id) for neuron_id in neuron_ids]
            first_input = input(f"Select a neuron id from: {neuron_ids}")
            if first_input == "":
                print("No neuron id selected, exiting")
                return None
            first_path = str(pi) + "_neuron_" + str(first_input)

            second_input = input(
                f"Select a file to load from: {ls(hdf5_file[first_path])}"
            )
            if second_input == "":
                print("No attribute selected, exiting")
                return None
            second_path = first_path + "/" + str(second_input)

            return hdf5_file[second_path][(...)]
        else:
            return_path = str(pi) + "_neuron_" + str(id) + "/" + str(file)
            return hdf5_file[return_path][(...)]


def ls(hdf5_file_path):
    """
    Given an hdf5 file path or an open hdf5 file python object, returns the child directories.
    """
    if type(hdf5_file_path) is str:
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            return list(hdf5_file.keys())
    else:
        return list(hdf5_file_path.keys())


def normalise_wf(wf):
    """
    Custom normalisation so that the through of the waveform is set to -1
    or the peak is set to +1 if the waveform is dendritic
    """
    baseline = wf[:, :20].mean(axis=1, keepdims=True)
    through = wf.min()
    peak = wf.max()
    if np.abs(through) > np.abs(peak):
        normalised = (wf - baseline) / np.abs(through)
    else:
        normalised = (wf - baseline) / np.abs(peak)
    return normalised


def preprocess(image, central_range=60, n_channels=10):
    """
    It takes an image, and returns a copy of the image with the central 60 pixels in the horizontal
    direction, and the central 10 pixels in the vertical direction

    Args:
      image: the image to be preprocessed
      central_range: the number of pixels to take from the centre of the image. Defaults to 60
      n_channels: The number of channels to use in the final image. Defaults to 10

    Returns:
      The image is being cropped to the central range and the number of channels specified.
    """
    peak_channel = image.shape[0] // 2
    centre = image.shape[1] // 2
    return image[
        (peak_channel - n_channels // 2) : (peak_channel + n_channels // 2),
        (centre - central_range // 2) : (centre + central_range // 2),
    ].copy()


def extract_raw_data(
    dataset,
    central_range,
    n_channels,
    raw_spikes=False,
    labels_only=True,
    n_unlab=0,
    quality_check=True,
    debug=False,
    json_path=None,
    verbose=False,
    normalise=True,
):
    """
    It takes a dataset, extracts the waveforms, acg and labels, and returns them as lists

    Args:
        dataset: the path to the hdf5 file containing the dataset
        central_range: the number of samples to extract from the central part of the waveform
        n_channels: number of channels to extract from the waveform
        raw_spikes: if True, returns the raw spike times instead of the ACG. If None, returns both.
        Defaults to False
        labels_only: if True, only labelled neurons are extracted. If False, also unlabelled ones are
        extracted, up to a maximum of n_unlab. Defaults to True
        n_unlab: number of unlabelled neurons to extract. Defaults to 0
        quality_check: if True, we check for sane spikes and fn/fp spikes. Defaults to True
        debug: if True, will plot the discarded neurons and their ACG and waveforms. Defaults to False
        json_path: path to the json file containing the dataset paths
        verbose: if True, prints out the reason why a neuron was discarded. Defaults to False
        normalise: if True, normalises the waveforms and the acg. Defaults to True
    """
    # Load dataset paths if in debug mode
    if debug == True and json_path is None:
        print("Need to provide raw dataset path for debugging purposes")
        return NotImplementedError
    elif debug == True:
        with open(json_path) as f:
            json_f = json.load(f)

        path_list = []
        path_name_list = []
        for ds in json_f.values():
            dp = Path(ds["dp"])
            path_list.append(dp)
            path_name_list.append(dp.name)

    wf_list = []
    acg_list = []
    spikes_list = []
    labels_list = []
    dataset_info = []
    count_non_lab = 0

    neuron_ids = []
    with h5py.File(dataset, "r") as hdf5_file:
        for name in hdf5_file:
            if "neuron" in name:
                neuron_id = name.split("_")[-1]
                neuron_ids.append(int(neuron_id))

    discarded_df = pd.DataFrame(columns=["neuron_id", "label", "dataset", "reason"])
    for wf_n in tqdm(np.sort(neuron_ids), desc="Reading dataset", leave=False):
        try:
            # Get the label for this wvf
            label = get_neuron_attr(dataset, wf_n, "optotagged_label").ravel()[0]

            # If the neuron is labelled we extract it anyways
            if label != 0:
                label = str(label.decode("utf-8"))
                labels_list.append(label)

            # else if it is unlabelled and we only care about labelled ones, skip this one
            elif (label == 0 or None) and labels_only == True:
                continue

            # Otherwise extract it, if we still did not reach the desired count of unlabelled ones
            elif (
                (label == 0 or None) and count_non_lab < n_unlab
            ) and labels_only == False:
                count_non_lab += 1
                labels_list.append("unlabelled")

            # Otherwise skip it, there might be other labelled ones ahead (which is why we don't break here)
            else:
                continue

            spike_idxes = get_neuron_attr(dataset, wf_n, "spike_indices")

            if quality_check:
                sane_spikes = get_neuron_attr(dataset, wf_n, "sane_spikes")
                fn_fp_spikes = get_neuron_attr(dataset, wf_n, "fn_fp_filtered_spikes")
                mask = fn_fp_spikes & sane_spikes
                spikes = spike_idxes[mask].copy()
            else:
                spikes = spike_idxes

            # if spikes is void after quality checks, skip this neuron
            if len(spikes) == 0:
                dataset_name = (
                    get_neuron_attr(dataset, wf_n, "dataset_id")
                    .ravel()[0]
                    .decode("utf-8")
                )
                if verbose:
                    print(
                        f"Discarded wf n. {wf_n} ({label}) after quality checks. Dataset: {dataset_name}"
                    )
                discarded_df = pd.concat(
                    (
                        discarded_df,
                        pd.DataFrame(
                            {
                                "neuron_id": [wf_n],
                                "label": [label],
                                "dataset": [dataset_name],
                                "reason": ["quality checks"],
                            }
                        ),
                    ),
                    ignore_index=True,
                )

                # Debugging options
                if debug == True:
                    dataset_idx = path_name_list.index(dataset_name)
                    dataset_path = path_list[dataset_idx]
                    npyx.spk_t.trn_filtered(
                        dataset_path,
                        get_neuron_attr(dataset, wf_n, "neuron_id").ravel()[0],
                        plot_debug=True,
                        again=1,
                    )
                    plt.show()
                    npyx.plot.plot_acg(
                        dataset_path,
                        get_neuron_attr(dataset, wf_n, "neuron_id").ravel()[0],
                        0.2,
                        80,
                    )
                    plt.show()
                    print("\n --------------------------------- \n")

                del labels_list[-1]
                if label == 0:
                    count_non_lab -= 1
                continue

            # Extract waveform using provided parameters
            wf = get_neuron_attr(dataset, wf_n, "mean_waveform_preprocessed")
            if normalise:
                wf_list.append(
                    preprocess(normalise_wf(wf), central_range, n_channels)
                    .ravel()
                    .astype(float)
                )
            else:
                wf_list.append(
                    preprocess(wf, central_range, n_channels).ravel().astype(float)
                )

            # Extract and normalise acg if requested
            if raw_spikes == False:
                if normalise:
                    # mean_fr = npyx.spk_t.mean_firing_rate(spikes)
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    normal_acg = np.clip(acg / np.max(acg), 0, 10)
                    acg_list.append(normal_acg.astype(float))
                else:
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    acg_list.append(acg.astype(float))

            elif raw_spikes == True:
                acg_list.append(spikes)

            # in this special case we extract both the acg and the spikes
            elif raw_spikes == None:
                if normalise:
                    # mean_fr = npyx.spk_t.mean_firing_rate(spikes)
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    normal_acg = np.clip(acg / np.max(acg), 0, 10)
                    acg_list.append(normal_acg.astype(float))
                    spikes_list.append(spikes.astype(int))
                else:
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    acg_list.append(acg.astype(float))
                    spikes_list.append(spikes.astype(int))

            # Extract useful metadata
            dataset_name = (
                get_neuron_attr(dataset, wf_n, "dataset_id").ravel()[0].decode("utf-8")
            )
            neuron_id = get_neuron_attr(dataset, wf_n, "neuron_id").ravel()[0]
            neuron_metadata = dataset_name + "/" + str(neuron_id)
            dataset_info.append(str(neuron_metadata))
        except KeyError as e:
            if verbose:
                print(f"KeyError for neuron {wf_n}, skipping...")
                print(f"Further details: {e.args[0]}")
            discarded_df = pd.concat(
                (
                    discarded_df,
                    pd.DataFrame(
                        {
                            "neuron_id": [wf_n],
                            "label": [label],
                            "dataset": [dataset_name],
                            "reason": ["KeyError"],
                        }
                    ),
                ),
                ignore_index=True,
            )
            continue
    print("Details of discarded neurons:")
    print(discarded_df)
    print(
        f"{len(discarded_df)} total neurons discarded, of which labelled: {len(discarded_df[discarded_df.label != 0])}"
    )

    if raw_spikes == None:
        return wf_list, acg_list, spikes_list, labels_list, dataset_info
    else:
        return wf_list, acg_list, labels_list, dataset_info


def replace(my_list, my_dict):
    return [x if x not in my_dict else my_dict[x] for x in my_list]


def plot_random_neuron(
    full_dataset, n_neurons=1, label=None, dataset_info=None, show=True
):
    """
    Plots a random neuron from the dataset. Optionally, can specify a label
    """
    if label is not None:
        if label in CORRESPONDENCE:
            mask = full_dataset[:, 0] == label
            full_dataset = full_dataset[mask, :]
            dataset_info = list(np.array(dataset_info)[mask])
        elif label in LABELLING:
            mask = full_dataset[:, 0] == LABELLING[label]
            full_dataset = full_dataset[mask, :]
            dataset_info = list(np.array(dataset_info)[mask])
        else:
            return "Invalid label"

    full_dataset = np.atleast_2d(full_dataset)

    if n_neurons > full_dataset.shape[0]:
        print(
            f"Only {full_dataset.shape[0]} neurons to plot with the provided parameter choice"
        )
        n_neurons = full_dataset.shape[0]

    idx = np.random.choice(range(full_dataset.shape[0]), n_neurons, replace=False)
    for i in idx:
        info = f" - Dataset: {dataset_info[i]}" if dataset_info is not None else ""
        fig, axs = plt.subplots(1, 4, figsize=(16, 3), constrained_layout=True)
        fig.suptitle(
            r"$\bf{"
            + CORRESPONDENCE[full_dataset[i, 0]].replace("_", "\_")
            + "}$ "
            + info
        )
        wf = full_dataset[i, 1 : CENTRAL_RANGE * N_CHANNELS + 1].reshape(
            N_CHANNELS, CENTRAL_RANGE
        )
        acg = full_dataset[i, CENTRAL_RANGE * N_CHANNELS + 1 :]

        axs[0].imshow(wf, interpolation="nearest", aspect="auto")
        rect = patches.Rectangle(
            (0, N_CHANNELS // 2 - 0.5),
            CENTRAL_RANGE - 1,
            1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        axs[0].add_patch(rect)
        axs[0].set_title("Raw waveform across channels")

        n_lines = wf.shape[0]
        x = range(wf.shape[1])
        linewidth = 4
        for i, row in enumerate(wf):
            if i == 5:
                line = row + (n_lines - i)
                axs[1].plot(x, line, lw=linewidth, c="red", alpha=1, zorder=i / n_lines)
            # elif i % 2 == 0:
            #     continue
            else:
                line = row + (n_lines - i)
                axs[1].plot(
                    x, line, lw=linewidth - 1, c="black", alpha=1, zorder=i / n_lines
                )
        axs[1].set_title("Raw waveforms")
        axs[1].set_yticks([])

        axs[2].plot(wf[N_CHANNELS // 2, :])
        axs[2].set_title("Peak channel waveform")

        axs[3].plot(acg)
        axs[3].set_title("Autocorrelogram")
        if show:
            plt.show()
        else:
            return fig


def vertical_reflection(waveform):
    """
    Vertically reflects the channels of a waveform
    """

    if waveform.shape != (N_CHANNELS, CENTRAL_RANGE):
        waveform = waveform.reshape(N_CHANNELS, CENTRAL_RANGE)
    new_waveform = waveform[::-1].copy()
    return new_waveform


def swap_channel(waveform):
    """
    Swaps the channels of a waveform
    """
    if waveform.shape != (N_CHANNELS, CENTRAL_RANGE):
        waveform = waveform.reshape(N_CHANNELS, CENTRAL_RANGE)
    evens = waveform[1::2, :]
    odds = waveform[::2, :]
    new_waveform = (
        np.array([(i, j) for i, j in zip(odds, evens)])
        .ravel()
        .reshape(N_CHANNELS, CENTRAL_RANGE)
    )
    return new_waveform


def resample_acg(acg, window_size=20, keep_same_size=True):
    """
    Given an ACG, add artificial points to it.
    If keep_same_size is True, the ACG will be of the same size: this is achieved
    by undersapling points at the end of the ACG.
    """
    y = np.array(acg).copy()
    X = np.linspace(0, len(y), len(y))

    interpolated_window = y[:window_size]
    # Create interpolating points
    avg_arr = (interpolated_window + np.roll(interpolated_window, -1)) / 2.0
    avg_enhanced = np.vstack([interpolated_window, avg_arr]).flatten("F")[:-1]

    # Create new_y enhanced with interpolating points
    new_y = np.concatenate((avg_enhanced.ravel(), y[window_size:].ravel()), axis=0)

    if keep_same_size == False:
        return new_y

    # Select final points to remove
    idxes = np.ones_like(new_y).astype(bool)
    idxes[-2 * window_size :: 2] = False

    return new_y[idxes]


def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
    seed : Integer
            A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Default is `True`.

    Returns:
    Nothing.
    """
    if seed is None:
        seed = np.random.choice(2**16)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")
    return seed


class NeuronsDataset:
    """
    Custom class for the cerebellum dataset, containing all infomation about the labelled and unlabelled neurons.
    """

    def __init__(
        self,
        dataset,
        quality_check=True,
        normalise=True,
        resample=True,
        central_range=CENTRAL_RANGE,
        n_channels=N_CHANNELS,
    ):
        self.wf_list = []
        self.acg_list = []
        self.spikes_list = []
        self.labels_list = []
        self.info = []

        neuron_ids = []
        with h5py.File(dataset, "r") as hdf5_file:
            for name in hdf5_file:
                if "neuron" in name:
                    neuron_id = name.split("_")[-1]
                    neuron_ids.append(int(neuron_id))

        discarded_df = pd.DataFrame(columns=["neuron_id", "label", "dataset", "reason"])
        for wf_n in tqdm(np.sort(neuron_ids), desc="Reading dataset", leave=False):
            try:
                # Get the label for this wvf
                label = get_neuron_attr(dataset, wf_n, "optotagged_label").ravel()[0]

                # If the neuron is labelled we extract it anyways
                if label != 0:
                    label = str(label.decode("utf-8"))
                    self.labels_list.append(label)

                # Otherwise extract it, if we still did not reach the desired count of unlabelled ones
                else:
                    self.labels_list.append("unlabelled")

                spike_idxes = get_neuron_attr(dataset, wf_n, "spike_indices")

                if quality_check:
                    sane_spikes = get_neuron_attr(dataset, wf_n, "sane_spikes")
                    fn_fp_spikes = get_neuron_attr(
                        dataset, wf_n, "fn_fp_filtered_spikes"
                    )
                    mask = fn_fp_spikes & sane_spikes
                    spikes = spike_idxes[mask].copy()
                else:
                    spikes = spike_idxes

                # if spikes is void after quality checks, skip this neuron
                if len(spikes) == 0:
                    dataset_name = (
                        get_neuron_attr(dataset, wf_n, "dataset_id")
                        .ravel()[0]
                        .decode("utf-8")
                    )
                    discarded_df = pd.concat(
                        (
                            discarded_df,
                            pd.DataFrame(
                                {
                                    "neuron_id": [wf_n],
                                    "label": [label],
                                    "dataset": [dataset_name],
                                    "reason": ["quality checks"],
                                }
                            ),
                        ),
                        ignore_index=True,
                    )
                    del self.labels_list[-1]
                    continue

                # Extract waveform using provided parameters
                wf = get_neuron_attr(dataset, wf_n, "mean_waveform_preprocessed")
                if normalise:
                    self.wf_list.append(
                        preprocess(normalise_wf(wf), central_range, n_channels)
                        .ravel()
                        .astype(float)
                    )
                else:
                    self.wf_list.append(
                        preprocess(wf, central_range, n_channels).ravel().astype(float)
                    )

                if normalise:
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    normal_acg = np.clip(acg / np.max(acg), 0, 10)
                    self.acg_list.append(normal_acg.astype(float))
                    self.spikes_list.append(spikes.astype(int))
                else:
                    acg = npyx.corr.acg("hello", 4, 1, 200, train=spikes)
                    self.acg_list.append(acg.astype(float))
                    self.spikes_list.append(spikes.astype(int))

                # Extract useful metadata
                dataset_name = (
                    get_neuron_attr(dataset, wf_n, "dataset_id")
                    .ravel()[0]
                    .decode("utf-8")
                )
                neuron_id = get_neuron_attr(dataset, wf_n, "neuron_id").ravel()[0]
                neuron_metadata = dataset_name + "/" + str(neuron_id)
                self.info.append(str(neuron_metadata))

            except KeyError as e:
                discarded_df = pd.concat(
                    (
                        discarded_df,
                        pd.DataFrame(
                            {
                                "neuron_id": [wf_n],
                                "label": [label],
                                "dataset": [dataset_name],
                                "reason": ["KeyError"],
                            }
                        ),
                    ),
                    ignore_index=True,
                )
                continue

        self.discarded_df = discarded_df

        acg_list_cut = [x[int(len(x) / 2) :] for x in self.acg_list]
        if resample is True:
            acg_list_resampled = list(map(resample_acg, acg_list_cut))
        else:
            acg_list_resampled = acg_list_cut

        self.targets = np.array((pd.Series(self.labels_list).replace(LABELLING).values))
        self.wf = np.stack(self.wf_list, axis=0)
        self.acg = np.stack(acg_list_resampled, axis=0)

        print(
            f"{len(self.wf_list)} neurons loaded, of which labelled: {sum(self.targets != -1)} \n"
            f"{len(discarded_df)} neurons discarded, of which labelled: {len(discarded_df[discarded_df.label != 0])}. More details at the 'discarded_df' attribute."
        )

    def make_labels_only(self):
        """
        It removes all the data points that have no labels
        """
        mask = self.targets != -1
        self.wf = self.wf[mask]
        self.acg = self.acg[mask]
        self.targets = self.targets[mask]
        self.info = np.array(self.info)[mask].tolist()
        self.spikes_list = np.array(self.spikes_list, dtype=object)[mask].tolist()

    def make_full_dataset(self, wf_only=False, acg_only=False):
        """
        > This function takes the waveform and ACG data and concatenates them into a single array

        Args:
            wf_only: If True, only the waveform data will be used. Defaults to False
            acg_only: If True, only the ACG data will be used. Defaults to False
        """
        if wf_only:
            self.full_dataset = self.wf
        elif acg_only:
            self.full_dataset = self.acg
        else:
            self.full_dataset = np.concatenate((self.wf, self.acg), axis=1)

    def min_max_scale(self, mean=False):
        """
        `min_max_scale` takes the waveform and ACG and scales them to the range [-1, 1] by dividing by the
        maximum absolute value of the waveform and ACG

        Args:
            mean: If True, the mean of the first 100 largest waveforms will be used as the scaling value.
            If False, the maximum value of the waveforms will be used. Defaults to False.
        """
        if mean:
            self._scale_value_wf = (np.sort(self.wf.ravel())[:100]).mean()
            self._scale_value_acg = (np.sort(self.acg.ravel())[-100:]).mean()
        else:
            self._scale_value_wf = np.max(np.abs(self.wf))
            self._scale_value_acg = np.max(np.abs(self.acg))
        self.wf = self.wf / self._scale_value_wf
        self.acg = self.acg / self._scale_value_acg


def plot_reconstruction(neurons_dataset: NeuronsDataset, VAE, n_neurons=1, label=None):
    """
    Plots a random neuron from the dataset. Optionally, can specify a label
    """

    plotting_dataset = copy.copy(neurons_dataset)
    plotting_dataset.info = np.array(plotting_dataset.info)

    if label is not None:
        if label in CORRESPONDENCE:
            mask = plotting_dataset.targets == label
            plotting_dataset.full_dataset = plotting_dataset.full_dataset[mask]
            plotting_dataset.info = plotting_dataset.info[mask]
            plotting_dataset.targets = plotting_dataset.targets[mask]
        elif label in LABELLING:
            mask = plotting_dataset.targets == LABELLING[label]
            plotting_dataset.full_dataset = plotting_dataset.full_dataset[mask]
            plotting_dataset.info = plotting_dataset.info[mask]
            plotting_dataset.targets = plotting_dataset.targets[mask]
        else:
            return "Invalid label"

    if n_neurons > plotting_dataset.full_dataset.shape[0]:
        print(
            f"Only {plotting_dataset.full_dataset.shape[0]} neurons to plot with the provided parameter choice"
        )
        n_neurons = plotting_dataset.full_dataset.shape[0]

    idx = np.random.choice(
        range(plotting_dataset.full_dataset.shape[0]), n_neurons, replace=False
    )
    for i in idx:
        fig, axs = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
        fig.suptitle(
            r"$\bf{"
            + CORRESPONDENCE[plotting_dataset.targets[i]]
            + "}$ - "
            + "Dataset: "
            + plotting_dataset.info[i]
        )
        wf_original = plotting_dataset.full_dataset[
            i, : N_CHANNELS * CENTRAL_RANGE
        ].reshape(N_CHANNELS, CENTRAL_RANGE)
        acg_original = plotting_dataset.full_dataset[i, N_CHANNELS * CENTRAL_RANGE :]

        axs[0, 0].imshow(wf_original, interpolation="nearest", aspect="auto")
        rect = patches.Rectangle(
            (0, N_CHANNELS // 2 - 0.5),
            CENTRAL_RANGE - 1,
            1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        axs[0, 0].add_patch(rect)
        axs[0, 0].set_title("Original raw waveform across channels")

        n_lines = wf_original.shape[0]
        x = range(wf_original.shape[1])
        linewidth = 4
        for j, row in enumerate(wf_original):
            if j == 5:
                line = row + (n_lines - j)
                axs[0, 1].plot(
                    x, line, lw=linewidth, c="red", alpha=1, zorder=j / n_lines
                )
            # elif i % 2 == 0:
            #     continue
            else:
                line = row + (n_lines - j)
                axs[0, 1].plot(
                    x, line, lw=linewidth, c="grey", alpha=1, zorder=j / n_lines
                )
        axs[0, 1].set_title("Original Raw waveforms")
        axs[0, 1].set_yticks([])

        axs[0, 2].plot(wf_original[N_CHANNELS // 2, :])
        axs[0, 2].set_title("Original Peak channel waveform")

        axs[0, 3].plot(np.concatenate((acg_original[::-1], acg_original)))
        axs[0, 3].set_title("Original autocorrelogram")

        # Getting this y scale for later:
        y_min, y_max = axs[0, 3].get_ylim()

        VAE = VAE.to("cpu")
        VAE.eval()
        with torch.no_grad():
            network_input = torch.Tensor(
                plotting_dataset.full_dataset[i, :].reshape(1, -1).astype(np.float32)
            ).to("cpu")
            all_reconstructed, all_reconstructed_stddev = VAE.reconstruct_img(
                network_input
            )
            all_reconstructed = all_reconstructed.detach().numpy().squeeze()
            all_reconstructed_stddev = (
                all_reconstructed_stddev.detach().numpy().squeeze()
            )
            #! Need to handle the case with only waveforms
            if all_reconstructed.shape[0] == N_CHANNELS * CENTRAL_RANGE:
                wf_reconstructed = all_reconstructed.reshape(N_CHANNELS, CENTRAL_RANGE)
                wf_reconstructed_stddev = all_reconstructed_stddev.reshape(
                    N_CHANNELS, CENTRAL_RANGE
                )

                reconstructed_acg = acg_original
                acg_reconstructed_stddev = np.zeros_like(reconstructed_acg)
            else:
                wf_reconstructed = all_reconstructed[
                    : N_CHANNELS * CENTRAL_RANGE
                ].reshape(N_CHANNELS, CENTRAL_RANGE)
                reconstructed_acg = all_reconstructed[N_CHANNELS * CENTRAL_RANGE :]
                wf_reconstructed_stddev = all_reconstructed_stddev[
                    : N_CHANNELS * CENTRAL_RANGE
                ].reshape(N_CHANNELS, CENTRAL_RANGE)
                acg_reconstructed_stddev = all_reconstructed_stddev[
                    N_CHANNELS * CENTRAL_RANGE :
                ]

        axs[1, 0].imshow(wf_reconstructed, interpolation="nearest", aspect="auto")
        rect = patches.Rectangle(
            (0, N_CHANNELS // 2 - 0.5),
            CENTRAL_RANGE - 1,
            1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        axs[1, 0].add_patch(rect)
        axs[1, 0].set_title("Reconstructed raw waveform across channels")

        n_lines = wf_reconstructed.shape[0]
        x = range(wf_reconstructed.shape[1])
        linewidth = 4
        for i, row in enumerate(wf_reconstructed):
            if i == 5:
                line = row + (n_lines - i)
                axs[1, 1].plot(
                    x, line, lw=linewidth, c="red", alpha=1, zorder=i / n_lines
                )
            # elif i % 2 == 0:
            #     continue
            else:
                line = row + (n_lines - i)
                axs[1, 1].plot(
                    x, line, lw=linewidth, c="grey", alpha=1, zorder=i / n_lines
                )
        axs[1, 1].set_title("Reconstructed Raw waveforms")
        axs[1, 1].set_yticks([])

        axs[1, 2].plot(wf_reconstructed[N_CHANNELS // 2, :])
        axs[1, 2].set_title("Reconstructed Peak channel waveform")
        axs[1, 2].fill_between(
            range(wf_reconstructed.shape[1]),
            wf_reconstructed[N_CHANNELS // 2, :]
            + wf_reconstructed_stddev[N_CHANNELS // 2, :],
            wf_reconstructed[N_CHANNELS // 2, :]
            - wf_reconstructed_stddev[N_CHANNELS // 2, :],
            facecolor="blue",
            alpha=0.2,
        )

        full_reconstructed_acg = np.concatenate(
            (reconstructed_acg[::-1], reconstructed_acg)
        )
        full_reconstructed_acg_stdev = np.concatenate(
            (acg_reconstructed_stddev[::-1], acg_reconstructed_stddev)
        )
        axs[1, 3].plot(full_reconstructed_acg)
        axs[1, 3].set_title("Reconstructed autocorrelogram")
        axs[1, 3].fill_between(
            range(full_reconstructed_acg.shape[0]),
            full_reconstructed_acg + full_reconstructed_acg_stdev,
            full_reconstructed_acg - full_reconstructed_acg_stdev,
            alpha=0.2,
        )
        axs[1, 3].set_ylim(y_min, y_max)

        plt.show()
