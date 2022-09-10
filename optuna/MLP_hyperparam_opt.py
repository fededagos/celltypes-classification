"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""
import os
import logging
import sys

import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm

import utils.h5_utils as h5
from utils.cerebellum import CerebellumDatasetSSL, RandTrans
import utils.custom_transforms as transforms
from utils.constants import DATA_PATH, N_CHANNELS, CENTRAL_RANGE, ACG_LEN


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 40
CLASSES = 6
DIR = os.getcwd()
EPOCHS = 50
N_TRAIN_EXAMPLES = None
N_VALID_EXAMPLES = None
BASE_DATASET = h5.NeuronsDataset(DATA_PATH, quality_check=True, normalise=False)
BASE_DATASET.min_max_scale()
BASE_DATASET.make_labels_only()
BASE_DATASET.make_full_dataset()


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = N_CHANNELS * CENTRAL_RANGE + ACG_LEN
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 20, 150)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_dataset(train_idx, val_idx):

    base_dataset = BASE_DATASET

    global N_TRAIN_EXAMPLES
    global N_VALID_EXAMPLES
    N_TRAIN_EXAMPLES = len(train_idx)
    N_VALID_EXAMPLES = len(val_idx)

    train_dataset = CerebellumDatasetSSL(
        base_dataset.full_dataset,
        base_dataset.targets,
        base_dataset.spikes_list,
        train_idx,
        transform=RandTrans(n=2, m=4),
    )

    val_dataset = CerebellumDatasetSSL(
        base_dataset.full_dataset,
        base_dataset.targets,
        base_dataset.spikes_list,
        val_idx,
        transform=None,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=True
    )

    return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)
    accuracies = []

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(BASE_DATASET.full_dataset, BASE_DATASET.targets)),
        leave=True,
        position=0,
        desc="Cross-validating",
        total=5,
    ):
        # Get the Cerebellum dataset.
        train_loader, valid_loader = get_dataset(train_idx, val_idx)
        # Training of the model.
        for epoch in tqdm(
            range(EPOCHS), desc="Training", position=1, leave=False, total=EPOCHS,
        ):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # Limiting training data for faster epochs.
                if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        fold_accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
        accuracies.append(fold_accuracy)
        trial.report(np.array(accuracies).mean(), fold)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.array(accuracies).mean()


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "example-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
