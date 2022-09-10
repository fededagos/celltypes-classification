import os
import logging
import sys
import inspect
from copy import copy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import itertools
import pandas as pd
import random
import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributions as dist
import torch.optim as optim
from torch.optim import Adam, RMSprop
import torch.utils.data
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import utils.models as m
from torchvision import transforms
from sklearn.model_selection import LeaveOneOut

import utils.h5_utils as h5
from utils.constants import DATA_PATH, N_CHANNELS, CENTRAL_RANGE, LABELLING
from pathlib import Path
from npyx.feat import temporal_features

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score

PATH = os.path.dirname(os.path.abspath(""))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = 6
DIR = os.getcwd()
BASE_DATASET = h5.NeuronsDataset(DATA_PATH, quality_check=True, normalise=False)
BASE_DATASET.min_max_scale()
BASE_DATASET.make_full_dataset(wf_only=True)
LABELS_ONLY_DATASET = copy(BASE_DATASET)
LABELS_ONLY_DATASET.make_labels_only()
SEED = 1234
USE_CUDA = torch.cuda.is_available()
TEST_FREQUENCY = 1
BETA = 1
INIT_WEIGHTS = True

BATCH_SIZE = 50
N_LOOS = 10

torch.set_default_dtype(torch.float32)


h5.set_seed(SEED)


best_encoder = None
best_decoder = None

enc = None
dec = None


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

        log_prior = prior.log_prob(z)  # log( p(z_i) )
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
        if i == 0:
            out_features = trial.suggest_int("n_units_l{}".format(i), 20, 200)
            p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5, step=0.01)
            first_units = out_features
        else:
            # Ensuring it is a valid VAE architecture otherwise the reconstructions look like nonsense!
            out_features = trial.suggest_int("n_units_l{}".format(i), 20, trial.params[f"n_units_l{i-1}"])
            p = trial.suggest_float("dropout_l{}".format(i), 0.1, 0.5, step=0.01)

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

    h5.set_seed(SEED)
    global enc
    global dec
    global BATCH_SIZE

    BATCH_SIZE = trial.suggest_int("batch_size", 10, 120)
    # Add augmentation transforms if wanted
    labels = BASE_DATASET.targets
    wf_dataset = BASE_DATASET.wf
    composed = transforms.Compose([m.VerticalReflection(p=0.3), m.SwapChannels(p=0.5)])
    cerebellum_dataset = CerebellumWFDataset(wf_dataset, labels, transform=composed)
    train_loader = data.DataLoader(
        cerebellum_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Generate the model.
    enc, dec = define_model(trial)

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

    #! Suggest beta
    beta = trial.suggest_float("beta", 1.0, 6.0, step=0.01)

    try:
        for epoch in tqdm(range(N_epochs), desc="Epochs"):
            train_loss = 0.0
            for (X, _) in train_loader:
                X = X.to(DEVICE)
                opt_vae.zero_grad()
                loss = -ELBO_VAE(enc, dec, X, beta=beta).mean()
                loss.backward()
                opt_vae.step()
                train_loss += loss.item() * X.shape[0] / len(cerebellum_dataset)
            scheduler.step()
            losses.append(train_loss)

    # Handling possible divergence
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    ################################################################################
    # Now we need to use the encoder to provide features to the random forest      #
    ################################################################################

    enc.eval()
    dec.eval()

    tmp_features = []
    for spikes in tqdm(LABELS_ONLY_DATASET.spikes_list):
        tmp_features.append(temporal_features(spikes))

    encoder_features = []
    for wf in tqdm(LABELS_ONLY_DATASET.wf):
        wf_tensor = (
            torch.tensor(wf, dtype=torch.float32)
            .to(DEVICE)
            .reshape(1, 1, N_CHANNELS, CENTRAL_RANGE)
        )
        with torch.no_grad():
            enc_features = enc(wf_tensor).mean.detach().cpu().numpy().ravel()
        encoder_features.append(enc_features)

    wf_dataset = np.stack(encoder_features, axis=0)
    tmp_dataset = np.stack(tmp_features, axis=0)

    X = np.concatenate((wf_dataset, tmp_dataset), axis=1)
    nan_mask = np.isnan(X).any(axis=1)
    X = X[~nan_mask]
    y = LABELS_ONLY_DATASET.targets[~nan_mask]

    h5.set_seed(SEED)

    forest_params = optuna.load_study(
        "random-forest-feat-eng", f"sqlite:///random-forest-feat-eng.db"
    ).best_params

    f1s = []
    for loo in tqdm(range(N_LOOS), desc="Leave one out runs", position=0):
        true_targets = []
        model_pred = []

        seed = np.random.choice(2**32)

        kfold = LeaveOneOut()
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(X, y)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=len(X),
        ):

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[val_idx]
            y_test = y[val_idx]

            oversample = RandomOverSampler(random_state=seed)

            X_big, y_big = oversample.fit_resample(X_train, y_train)

            model = RandomForestClassifier(**forest_params, random_state=seed)

            model.fit(X_big, y_big)
            pred = model.predict(X_test)

            true_targets.append(y_test)
            model_pred.append(pred)

        f1 = f1_score(true_targets, model_pred, average="macro")
        f1s.append(f1)
        trial.report(f1, loo)

    return np.array(f1s).mean()


def save_models(study, trial):
    global best_encoder
    global best_decoder
    if study.best_trial == trial:
        best_encoder = enc
        best_decoder = dec
        torch.save(best_encoder.state_dict(), "best_enc_v4.pt")
        torch.save(best_decoder.state_dict(), "best_dec_v4.pt")


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "true-random-beta_v4"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )

    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    study.enqueue_trial({'n_layers': 2})
    
    # study.enqueue_trial(
    #     {
    #         "d_latent": 10,
    #         "dropout_l0": 0.1,
    #         "dropout_l1": 0.1,
    #         "lr": 0.003,
    #         "n_layers": 2,
    #         "n_units_l0": 200,
    #         "n_units_l1": 100,
    #         "optimizer": "Adam",
    #         "beta": 1.0,
    #         "batch_size": 50,
    #     }
    # )

    # study.enqueue_trial(
    #     {
    #         "d_latent": 10,
    #         "dropout_l0": 0.1,
    #         "dropout_l1": 0.1,
    #         "lr": 0.001,
    #         "n_layers": 2,
    #         "n_units_l0": 200,
    #         "n_units_l1": 100,
    #         "optimizer": "Adam",
    #         "beta": 5.0,
    #         "batch_size": 50,
    #     }
    # )

    # study.enqueue_trial(
    #     {
    #         "beta": 5.13,
    #         "d_latent": 7,
    #         "dropout_l0": 0.22,
    #         "lr": 0.000274653336119621,
    #         "n_layers": 1,
    #         "n_units_l0": 129,
    #         "optimizer": "Adam",
    #         "batch_size": 50,
    #     }
    # )

    # study.enqueue_trial(
    #     {
    #         "batch_size": 83,
    #         "beta": 1.23,
    #         "d_latent": 13,
    #         "dropout_l0": 0.11,
    #         "dropout_l1": 0.14,
    #         "lr": 0.0016187879370748587,
    #         "n_layers": 2,
    #         "n_units_l0": 197,
    #         "n_units_l1": 195,
    #         "optimizer": "Adam",
    #     }
    # )

    study.optimize(objective, n_trials=300, timeout=None, callbacks=[save_models])

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
