"""
Model tryout with the best hyperparameters found by optuna
"""
import os
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import utils.h5_utils as h5
from utils.cerebellum import CerebellumDatasetSSL, RandTrans
import utils.custom_transforms as transforms
from utils.constants import (
    DATA_PATH,
    N_CHANNELS,
    CENTRAL_RANGE,
    ACG_LEN,
    CORRESPONDENCE,
)
from sklearn.model_selection import LeaveOneOut


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
        train_dataset, batch_size=BATCHSIZE, shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False
    )

    return train_loader, valid_loader


def train():

    # Generate the model.
    model = nn.Sequential(
        nn.Linear(N_CHANNELS * CENTRAL_RANGE + ACG_LEN, 117),
        nn.ReLU(),
        nn.Dropout(0.223957828021243),
        nn.Linear(117, 148),
        nn.ReLU(),
        nn.Dropout(0.401563000009004),
        nn.Linear(148, CLASSES),
        nn.LogSoftmax(dim=1),
    ).to(DEVICE)

    # Generate the optimizers.
    lr = 0.004265516314362439
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    accuracies = []
    loo = LeaveOneOut()
    for i, (train_idx, val_idx) in tqdm(
        enumerate(loo.split(BASE_DATASET.full_dataset)),
        desc="LOO",
        total=len(BASE_DATASET.full_dataset),
        leave=True,
        position=0,
    ):
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

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)
        accuracies.append(accuracy)
        if accuracy == 0:
            print(
                f"Mistake on trial {i + 1}. True label: {CORRESPONDENCE[int(target.item())]}, predicted: {CORRESPONDENCE[int(pred.item())]}"
            )
        print(f"LOO step: {i+1} | Mean LOO accuracy: {np.array(accuracies).mean():.4f}")
    return accuracy


def main():

    train()


if __name__ == "__main__":
    main()
