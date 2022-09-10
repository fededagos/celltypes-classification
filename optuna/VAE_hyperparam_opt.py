import inspect
import logging
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import itertools
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data as data
import utils.h5_utils as h5
import utils.models as m
from sklearn.model_selection import train_test_split
from torch.optim import Adam, RMSprop
from torchvision import transforms
from tqdm.auto import tqdm
from utils.constants import BATCH_SIZE, CENTRAL_RANGE, DATA_PATH, LABELLING, N_CHANNELS

import optuna
from optuna.trial import TrialState

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = 6
DIR = os.getcwd()
BASE_DATASET = h5.NeuronsDataset(DATA_PATH, quality_check=True, normalise=False)
BASE_DATASET.min_max_scale()
BASE_DATASET.make_full_dataset(wf_only=True)
SEED = 1234
USE_CUDA = torch.cuda.is_available()
TEST_FREQUENCY = 1
BETA = 1
INIT_WEIGHTS = True

torch.set_default_dtype(torch.float32)


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


labels = BASE_DATASET.targets
wf_dataset = BASE_DATASET.wf


class CerebellumWFDataset(data.Dataset):
    """Dataset of waveforms as images. Every batch will have shape:
    (batch_size, 1, N_CHANNELS, CENTRAL_RANGE)"""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (ndarray): Array of data points
            labels (string): Array of labels for the provided data
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = (
            self.data[idx, :].astype("float32").reshape(1, N_CHANNELS, CENTRAL_RANGE)
        )
        label = self.labels[idx].astype("int")

        sample = (data_point, label)

        if self.transform:
            sample = self.transform(sample)

        return sample


# Add augmentation transforms if wanted
composed = transforms.Compose([m.VerticalReflection(p=0.3), m.SwapChannels(p=0.5)])
cerebellum_dataset = CerebellumWFDataset(wf_dataset, labels, transform=composed)
train_loader = data.DataLoader(cerebellum_dataset, batch_size=BATCH_SIZE, shuffle=True)


def ELBO_VAE(enc, dec, X, beta=1):
    """

    INPUT:
    enc : Instance of `Encoder` class, which returns a distribution
          over Z when called on a batch of inputs X
    dec : Instance of `Decoder` class, which returns a distribution
          over X when called on a batch of inputs Z
    X   : A batch of datapoints, torch.FloatTensor of shape = (batch_size, 1, 28, 28).

    """

    batch_size = X.shape[0]
    n_samples = 20  # number of monte carlo samples
    prior = dist.MultivariateNormal(
        loc=torch.zeros(enc.d_latent).to(DEVICE),
        covariance_matrix=torch.eye(enc.d_latent).to(DEVICE),
    )
    ELBO = torch.zeros(batch_size).to(DEVICE)

    for _ in range(n_samples):

        q_z = enc.forward(X)  # q(Z | X)
        z = (
            q_z.rsample()
        )  # Samples from the encoder posterior q(Z | X) using the reparameterization trick
        p_x = dec.forward(z)  # distribution p(x | z)

        log_prior = prior.log_prob(z).to(DEVICE)  # log( p(z_i) )
        dec_log_likelihood = (
            p_x.log_prob(X).reshape(batch_size, -1).sum(axis=1)
        )  # log( p(x_i | z_i) )
        enc_posterior = q_z.log_prob(z).sum(axis=1)  # log( q(z_i | x_i) )

        # Calculate the ELBO on the whole batch
        ELBO += dec_log_likelihood + beta * (-enc_posterior + log_prior)

    return ELBO / n_samples


def define_model(trial: optuna.trial.Trial):

    n_layers = trial.suggest_int("n_layers", 1, 3)
    d_latent = trial.suggest_int("d_latent", 5, 20)
    encoder_layers = []
    decoder_layers = []
    first_units = None

    in_features = N_CHANNELS * CENTRAL_RANGE
    for i in range(n_layers):

        out_features = trial.suggest_int("n_units_l{}".format(i), 20, 200)
        p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5, step=0.01)
        if i == 0:
            first_units = out_features

        # Create and properly init encoder layer
        cur_enc_layer = nn.Linear(in_features, out_features)
        if INIT_WEIGHTS:
            cur_enc_layer.weight.data.normal_(0, 0.001)
            cur_enc_layer.bias.data.normal_(0, 0.001)

        # Create and properly init decoder layer
        cur_dec_layer = nn.Linear(out_features, in_features)
        if INIT_WEIGHTS:
            cur_dec_layer.weight.data.normal_(0, 0.001)
            cur_dec_layer.bias.data.normal_(0, 0.001)

        encoder_layers.append(cur_enc_layer)
        decoder_layers.append(cur_dec_layer)

        encoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Dropout(p))

        encoder_layers.append(nn.Dropout(p))
        decoder_layers.append(nn.ReLU())

        in_features = out_features
    encoder_layers.append(nn.Linear(in_features, d_latent))
    decoder_layers.append(nn.Linear(d_latent, in_features))

    encoder = nn.Sequential(*encoder_layers[:-1], nn.Linear(in_features, 2 * d_latent))
    decoder = nn.Sequential(
        *decoder_layers[:0:-1], nn.Linear(first_units, 2 * (N_CHANNELS * CENTRAL_RANGE))
    )

    encoder = Encoder(encoder, d_latent)
    decoder = Decoder(decoder, d_latent)

    return encoder.to(DEVICE), decoder.to(DEVICE)


class Encoder(nn.Module):
    def __init__(self, encoder, d_latent):
        super().__init__()
        self.encoder = encoder.float()
        self.d_latent = d_latent

    def forward(self, x):
        # flatten the image
        x = x.view(x.shape[0], -1)
        # forward pass through encoder network
        h = self.encoder(x)
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return dist.Normal(mu, torch.exp(log_var))


class Decoder(nn.Module):
    def __init__(self, decoder, d_latent):
        super().__init__()
        self.decoder = decoder.float()
        self.d_latent = d_latent

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        # forward pass through decoder network
        h = self.decoder(z)

        X_hat = h[:, : N_CHANNELS * CENTRAL_RANGE]
        log_sig = h[:, N_CHANNELS * CENTRAL_RANGE :]

        return dist.Normal(
            X_hat.reshape(-1, 1, N_CHANNELS, CENTRAL_RANGE),
            log_sig.reshape(-1, 1, N_CHANNELS, CENTRAL_RANGE).exp(),
        )


def objective(trial: optuna.trial.Trial):

    # Generate the model.
    torch.cuda.empty_cache()
    enc, dec = define_model(trial)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optim_args = {
        "params": itertools.chain(enc.parameters(), dec.parameters()),
        "lr": lr,
    }
    opt_vae = getattr(optim, optimizer_name)(**optim_args)

    # Add a scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        opt_vae, milestones=[90, 110, 130], gamma=0.5
    )

    N_epochs = 150
    losses = []

    my_path = os.path.dirname(os.path.abspath(__file__))

    plt.plot(losses)
    plt.title(f"Loss landscape for trial {trial.number}")
    plt.savefig(os.path.join(my_path, f"plots/{trial.number}_loss.png"))
    try:
        for epoch in tqdm(range(N_epochs), desc="Epochs"):
            train_loss = 0.0
            for (X, _) in train_loader:
                X = X.to(DEVICE)
                opt_vae.zero_grad()
                loss = -ELBO_VAE(enc, dec, X, beta=1).mean()
                loss.backward()
                opt_vae.step()
                train_loss += loss.item() * X.shape[0] / len(cerebellum_dataset)
            scheduler.step()
            losses.append(train_loss)

            trial.report(train_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    except ValueError:
        # Prune the trial in case of divergence
        raise optuna.exceptions.TrialPruned()

    plt.plot(losses)
    plt.title(f"Loss landscape for trial {trial.number}")
    plt.savefig(os.path.join(my_path, f"plots/{trial.number}_loss.png"))

    return losses[-1]


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "VAE-architecture"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize",
    )

    study.enqueue_trial(
        {
            "d_latent": 10,
            "dropout_l0": 0.1,
            "dropout_l1": 0.1,
            "lr": 0.003,
            "n_layers": 2,
            "n_units_l0": 200,
            "n_units_l1": 100,
            "optimizer": "Adam",
        }
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
