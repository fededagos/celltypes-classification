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
import utils.acg_models as m2
from torchvision import transforms
from sklearn.model_selection import LeaveOneOut

import utils.h5_utils as h5
from utils.constants import (
    DATA_PATH,
    N_CHANNELS,
    CENTRAL_RANGE,
    BATCH_SIZE,
    LABELLING,
    ACG_LEN,
)
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

N_LOOS = 10

torch.set_default_dtype(torch.float32)


h5.set_seed(SEED)


labels = BASE_DATASET.targets
wf_dataset = BASE_DATASET.wf
acg_dataset = BASE_DATASET.acg
spikes = BASE_DATASET.spikes_list

best_acg_encoder = None
best_acg_decoder = None

acg_enc = None
acg_dec = None


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


class CerebellumACGDataset(data.Dataset):

    """Dataset of ACGS. Every batch will have shape:
    (batch_size, 1, ACG_LEN)"""

    def __init__(self, data, labels, spikes, transform=None):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            labels (string): Array of labels for the provided data
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.spikes = spikes

    def __len__(self):
        return len(self.data)

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


# Add augmentation transforms if wanted
composed_wvf = transforms.Compose([m.VerticalReflection(p=0.3), m.SwapChannels(p=0.5)])
composed_acg = m2.CustomCompose(
    [
        m2.ConstantShift(p=0.1),
        m2.DeleteSpikes(p=0.01),
        m2.AddSpikes(p=0.01),
        m2.MoveSpikes(p=0.1),
    ]
)


cerebellum_wvf_dataset = CerebellumWFDataset(wf_dataset, labels, transform=composed_wvf)
train_loader_wvf = data.DataLoader(
    cerebellum_wvf_dataset, batch_size=BATCH_SIZE, shuffle=True
)

cerebellum_acg_dataset = CerebellumACGDataset(
    acg_dataset, labels, spikes, transform=composed_acg
)
train_loader_acg = data.DataLoader(
    cerebellum_acg_dataset, batch_size=BATCH_SIZE, shuffle=True
)


def ELBO_VAE(enc, dec, X, beta=1):
    """

    INPUT:
    enc : Instance of `Encoder` class, which returns a distribution
          over Z when called on a batch of inputs X
    dec : Instance of `Decoder` class, which returns a distribution
          over X when called on a batch of inputs Z
    X   : A batch of datapoints, torch.FloatTensor of shape = (batch_size, 1, 10, 60).

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


def define_wvf_model():

    best_params = optuna.load_study(
        "true-random-beta_v4", f"sqlite:///true-random-beta_v4.db"
    ).best_params

    n_layers = best_params["n_layers"]
    d_latent = best_params["d_latent"]

    in_features = N_CHANNELS * CENTRAL_RANGE

    encoder_layers = []
    decoder_layers = []
    first_units = None

    for i in range(n_layers):
        out_features = best_params[f"n_units_l{i}"]
        p = best_params[f"dropout_l{i}"]
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
    decoder = wvf_Decoder(decoder, d_latent)

    return encoder.to(DEVICE), decoder.to(DEVICE)


def define_acg_model(trial: optuna.trial.Trial):

    n_layers = trial.suggest_int("acg_n_layers", 1, 3)
    d_latent = trial.suggest_int("acg_d_latent", 5, 20)
    encoder_layers = []
    decoder_layers = []
    first_units = None

    in_features = ACG_LEN
    for i in range(n_layers):

        if i == 0:
            out_features = trial.suggest_int("acg_n_units_l{}".format(i), 20, 200)
            p = trial.suggest_float("acg_dropout_l{}".format(i), 0.1, 0.5, step=0.01)
            first_units = out_features
        else:
            # Ensuring it is a valid VAE architecture otherwise the reconstructions look like nonsense!
            out_features = trial.suggest_int(
                "acg_n_units_l{}".format(i), 20, trial.params[f"acg_n_units_l{i-1}"]
            )
            p = trial.suggest_float("acg_dropout_l{}".format(i), 0.1, 0.5, step=0.01)

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
        *decoder_layers[:0:-1], nn.Linear(first_units, 2 * (ACG_LEN))
    )

    encoder = Encoder(encoder, d_latent)
    decoder = acg_Decoder(decoder, d_latent)

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


class wvf_Decoder(nn.Module):
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


class acg_Decoder(nn.Module):
    def __init__(self, decoder, d_latent):
        super().__init__()
        self.decoder = decoder.float()
        self.d_latent = d_latent

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        # forward pass through decoder network
        h = self.decoder(z)

        X_hat = h[:, :ACG_LEN]
        log_sig = h[:, ACG_LEN:]

        return dist.Normal(
            X_hat.reshape(-1, 1, ACG_LEN),
            log_sig.reshape(-1, 1, ACG_LEN).exp(),
        )


def objective(trial: optuna.trial.Trial):

    h5.set_seed(SEED)
    wvf_enc, _ = define_wvf_model()

    wvf_enc.load_state_dict(torch.load("best_enc_v4.pt", map_location=DEVICE))

    ################################################################
    # Now same but for acg
    ################################################################

    global acg_enc
    global acg_dec

    # Generate the model.
    torch.cuda.empty_cache()
    acg_enc, acg_dec = define_acg_model(trial)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("acg_lr", 1e-5, 1e-1, log=True)
    optim_args = {
        "params": itertools.chain(acg_enc.parameters(), acg_dec.parameters()),
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
    beta_acg = trial.suggest_float("beta_acg", 1.0, 6.0, step=0.01)

    try:
        for epoch in tqdm(range(N_epochs), desc="Training ACG VAE"):
            train_loss = 0.0
            for (X, _) in train_loader_acg:
                X = X.to(DEVICE)
                opt_vae.zero_grad()
                loss = -ELBO_VAE(acg_enc, acg_dec, X, beta=beta_acg).mean()
                loss.backward()
                opt_vae.step()
                train_loss += loss.item() * X.shape[0] / len(cerebellum_acg_dataset)
            scheduler.step()
            losses.append(train_loss)

    # Handling possible divergence
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    ################################################################################
    # Now we need to use the encoder to provide features to the random forest      #
    ################################################################################

    # First setting all models to evaluation mode
    wvf_enc.eval()
    acg_enc.eval()

    tmp_features = []
    for acg in tqdm(LABELS_ONLY_DATASET.acg):
        acg_tensor = (
            torch.tensor(acg, dtype=torch.float32).to(DEVICE).reshape(1, 1, ACG_LEN)
        )
        with torch.no_grad():
            enc_features = acg_enc(acg_tensor).mean.detach().cpu().numpy().ravel()
        tmp_features.append(enc_features)

    wvf_features = []
    for wf in tqdm(LABELS_ONLY_DATASET.wf):
        wf_tensor = (
            torch.tensor(wf, dtype=torch.float32)
            .to(DEVICE)
            .reshape(1, 1, N_CHANNELS, CENTRAL_RANGE)
        )
        with torch.no_grad():
            enc_features = wvf_enc(wf_tensor).mean.detach().cpu().numpy().ravel()
        wvf_features.append(enc_features)

    wf_dataset = np.stack(wvf_features, axis=0)
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
    for loo in tqdm(range(N_LOOS), desc="Leave one out runs", leave=True, position=0):
        true_targets = []
        model_pred = []

        seed = np.random.choice(2**32)

        kfold = LeaveOneOut()
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(X, y)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=5,
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


def save_models(study: optuna.study.Study, trial):
    global best_acg_encoder
    global best_acg_decoder

    if study.best_trial == trial:
        best_acg_encoder = acg_enc
        best_acg_decoder = acg_dec
        torch.save(best_acg_encoder.state_dict(), "best_acg_enc_v2.pt")
        torch.save(best_acg_decoder.state_dict(), "best_acg_dec_v2.pt")


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "double-VAE_v2"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )

    study.enqueue_trial(
        {
            "acg_d_latent": 10,
            "acg_dropout_l0": 0.1,
            "acg_dropout_l1": 0.1,
            "lr": 0.003,
            "acg_n_layers": 2,
            "acg_n_units_l0": 200,
            "acg_n_units_l1": 100,
            "optimizer": "Adam",
            "beta_acg": 1.0,
        }
    )

    study.enqueue_trial(
        {
            "acg_d_latent": 10,
            "acg_dropout_l0": 0.1,
            "acg_dropout_l1": 0.1,
            "lr": 0.001,
            "acg_n_layers": 2,
            "acg_n_units_l0": 200,
            "acg_n_units_l1": 100,
            "optimizer": "Adam",
            "beta_acg": 5.0,
        }
    )
    study.enqueue_trial(
        {
            "acg_d_latent": 5,
            "acg_dropout_l0": 0.3,
            "lr": 0.001,
            "acg_n_layers": 1,
            "acg_n_units_l0": 200,
            "optimizer": "Adam",
            "beta_acg": 5.0,
        }
    )

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
