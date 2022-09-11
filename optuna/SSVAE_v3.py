# Adapted from https://pyro.ai/examples/ss-vae.html
import os
import sys
import inspect


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import npyx
import h5py
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import utils.h5_utils as h5
import utils.datasets as datasets
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pyro
import pyro.distributions as dist

from tqdm.auto import tqdm
from pathlib import Path
from visdom import Visdom
import pyro.optim as optim
from pyro.optim import Adam, RMSprop
from torchvision import transforms
from pyro.contrib.examples.util import print_and_log
from pyro_utils.custom_mlp import MLP, Exp

import optuna
from optuna.trial import TrialState
import logging

from pyro.infer import (
    SVI,
    JitTrace_ELBO,
    JitTraceEnum_ELBO,
    Trace_ELBO,
    TraceEnum_ELBO,
    config_enumerate,
)

from utils.constants import (
    DATA_PATH,
    N_CHANNELS,
    CENTRAL_RANGE,
    LABELLING,
    CORRESPONDENCE,
    ACG_LEN,
)


BATCH_SIZE = 30

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_CLASSES = 6

SEED = 1234

torch.set_default_dtype(torch.float32)

h5.set_seed(SEED)

dataset = h5.NeuronsDataset(DATA_PATH, normalise=False)

# Choosing the scaling and the type of dataset we want (full, acg or wf)
dataset.min_max_scale()
dataset.make_full_dataset()

full_dataset = dataset.full_dataset

composed = datasets.CustomCompose(
    [
        datasets.VerticalReflection(p=0.3),
        datasets.SwapChannels(p=0.5),
        datasets.AddSpikes(p=0.3, max_addition=0.15),
        datasets.DeleteSpikes(p=0.3, deletion_prob=0.1),
        datasets.MoveSpikes(p=0.2, max_shift=10),
        datasets.GaussianNoise(p=0.2),
        datasets.ConstantShift(p=0.3),
    ]
)


class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the cerebellum dataset

    :param output_size: size of the tensor representing the class label (6 for cerebellum since
                        we represent the class labels as a one-hot vector with 6 components)
    :param input_size: size of the tensor representing the image (10*60+100 = 700 for the cerebellum dataset
                       since we flatten the waveforms and concatenate the ACG)
    :param z_dim: size of the tensor representing the latent random variable z
                  (conceivably type of waveform and ACG)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """

    def __init__(
        self,
        output_size=N_CLASSES,
        input_size=N_CHANNELS * CENTRAL_RANGE + ACG_LEN,
        z_dim=50,
        hidden_layers=[
            500,
        ],
        hidden_layers_classifier=None,
        config_enum=None,
        use_cuda=False,
        aux_loss_multiplier=None,
        non_linearity=None,
    ):

        super().__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == "parallel"
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier
        self.non_linearity = non_linearity if non_linearity is not None else nn.ReLU
        self.hidden_layers_classifier = (
            hidden_layers
            if hidden_layers_classifier is None
            else hidden_layers_classifier
        )

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        hidden_sizes_classifier = self.hidden_layers_classifier

        # define the neural networks used later in the model and the guide.
        self.encoder_y = MLP(
            [self.input_size] + hidden_sizes_classifier + [self.output_size],
            activation=self.non_linearity,
            output_activation=nn.Softmax,
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = MLP(
            [self.input_size + self.output_size] + hidden_sizes + [[z_dim, z_dim]],
            activation=self.non_linearity,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )
        #! Change here for Gaussian output
        self.decoder = MLP(
            [z_dim + self.output_size]
            + hidden_sizes[::-1]
            + [[self.input_size, self.input_size]],
            activation=self.non_linearity,
            output_activation=[None, Exp],
            allow_broadcast=self.allow_broadcast,
            use_cuda=self.use_cuda,
        )

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # Waveform and ACG style
        p(y|x) = categorical(I/10.)     # which cell type
        p(x|y,z) = Normal(loc(y,z))     # a vector containing the ACG and waveform in space
        loc is given by a neural network  `decoder`

        :param xs: a batch of scaled vectors of concatenated ACG and waveform
        :param ys: (optional) a batch of the class labels i.e.
                   the cell type corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        options = dict(dtype=xs.dtype, device=xs.device)
        with pyro.plate("data"):

            # sample the latent from the constant prior distribution
            prior_loc = torch.zeros(batch_size, self.z_dim, **options)
            prior_scale = torch.ones(batch_size, self.z_dim, **options)
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = torch.ones(batch_size, self.output_size, **options) / (
                1.0 * self.output_size
            )
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)

            # Finally, score the x using the latent (z) and
            # the class label y against the
            # parametrized distribution p(x|y,z) = Normal(decoder(y,z))
            # where `decoder` is a neural network.
            loc, scale = self.decoder.forward([zs, ys])

            pyro.sample(
                "x", dist.Normal(loc, scale, validate_args=False).to_event(1), obs=xs
            )
            # return the loc and scale so we can visualize it later
            return loc, scale

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer cell type from vector
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer style from a vector and the cell type
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of scaled vectors of concatenated waveforms and ACGs
        :param ys: (optional) a batch of the class labels i.e.
                   the cell types corresponding to the vector(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):

            # if the class label is not supervised, sample
            # (and score) the cell type with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample (and score) the latent with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder_y.forward(xs)

        # get the index (label) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the label(s) to one-hot tensor(s)
        ys = torch.zeros_like(alpha).scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


def run_inference_for_epoch(
    data_loaders, losses, periodic_interval_batches, scheduler=None
):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.0] * num_losses
    epoch_losses_unsup = [0.0] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys) = next(unsup_iter)
        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    if scheduler is not None:
        scheduler.step()
    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    with torch.no_grad():
        predictions, actuals = [], []

        # use the appropriate data loader
        for (xs, ys) in data_loader:
            # use classification function to compute all predictions for each batch
            predictions.append(classifier_fn(xs))
            actuals.append(ys)

        # compute the number of accurate predictions
        accurate_preds = 0
        for pred, act in zip(predictions, actuals):
            for i in range(pred.size(0)):
                v = torch.sum(pred[i] == act[i])
                accurate_preds += v.item() == N_CLASSES

        # calculate the accuracy between 0 and 1
        accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    return accuracy


JIT = False
SEED = 1234
VISUALIZE = False
D_LATENT = 50
ETA = 1e-3
BETA = 0.9
CUDA = False
LOG = "./tmp.log"
N_SUPERVISED = 24
EPOCHS = 610
AUX_LOSS = True
INPUT_SIZE = N_CHANNELS * CENTRAL_RANGE + ACG_LEN
OUTPUT_SIZE = N_CLASSES


def objective(trial: optuna.trial.Trial):

    h5.set_seed(SEED)

    # Set up encoder and decoder variables

    D_LATENT = trial.suggest_int("d_latent", 5, 30)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    ETA = trial.suggest_float("acg_lr", 1e-5, 1e-1, log=True)
    NON_LINEARITY = trial.suggest_categorical("non_linearity", ["relu", "silu", "tanh"])
    AUX_MULTIPLIER = trial.suggest_int("aux_multiplier", 1, 99)
    BATCH_SIZE = trial.suggest_int("batch_size", 10, 120)

    N_LAYERS_VAE = trial.suggest_int("n_layers_vae", 1, 2)
    if N_LAYERS_VAE == 1:
        hidden_units_1 = trial.suggest_int("hidden_units_l1", 10, 200)
        HIDDEN_LAYERS = [
            hidden_units_1,
        ]

    elif N_LAYERS_VAE == 2:
        hidden_units_1 = trial.suggest_int("hidden_units_l1", 10, 200)
        hidden_units_2 = trial.suggest_int(
            "hidden_units_l2", 10, trial.params["hidden_units_l1"]
        )
        HIDDEN_LAYERS = [hidden_units_1, hidden_units_2]

    # Set up classifier layers
    N_LAYERS_CLASSIFIER = trial.suggest_int("n_layers_classifier", 1, 2)
    if N_LAYERS_CLASSIFIER == 1:
        hidden_units_1_class = trial.suggest_int("hidden_units_class_l1", 10, 200)
        HIDDEN_LAYERS_CLASSIFIER = [
            hidden_units_1_class,
        ]

    elif N_LAYERS_CLASSIFIER == 2:
        hidden_units_1_class = trial.suggest_int("hidden_units_class_l1", 10, 200)
        hidden_units_2_class = trial.suggest_int("hidden_units_class_l2", 10, 200)
        HIDDEN_LAYERS_CLASSIFIER = [hidden_units_1_class, hidden_units_2_class]

    if NON_LINEARITY == "relu":
        non_lin = nn.ReLU
    elif NON_LINEARITY == "silu":
        non_lin = nn.SiLU
    elif NON_LINEARITY == "tanh":
        non_lin = nn.Tanh

    pyro.clear_param_store()

    if SEED is not None:
        pyro.set_rng_seed(SEED)

    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = SSVAE(
        z_dim=D_LATENT,
        hidden_layers=HIDDEN_LAYERS,
        hidden_layers_classifier=HIDDEN_LAYERS_CLASSIFIER,
        input_size=INPUT_SIZE,
        use_cuda=CUDA,
        config_enum="parallel",
        aux_loss_multiplier=AUX_MULTIPLIER,
        output_size=OUTPUT_SIZE,
        non_linearity=non_lin,
    )
    if optimizer_name == "Adam":
        optim_args = {"lr": ETA, "betas": (BETA, 0.999)}
        optimizer = Adam(optim_args)
    else:
        optim_args = {
            "lr": ETA,
        }
        optimizer = getattr(optim, optimizer_name)(optim_args)

    # setup the optimizer

    scheduler = pyro.optim.CosineAnnealingWarmRestarts(
        {
            "optimizer": optimizer,
            "optim_args": optim_args,
            "T_0": 20,
            "T_mult": 2.0,
        }
    )

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    guide = config_enumerate(ss_vae.guide, "parallel", expand=True)
    Elbo = JitTraceEnum_ELBO if JIT else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=1, strict_enumeration_warning=False)
    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if AUX_LOSS:
        elbo = JitTrace_ELBO() if JIT else Trace_ELBO()
        loss_aux = SVI(
            ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo
        )
        losses.append(loss_aux)

    try:
        # setup the logger if a filename is provided
        logger = open(LOG, "w") if LOG else None

        data_loaders = datasets.setup_data_loaders(
            datasets.PyroCerebellumDataset,
            CUDA,
            BATCH_SIZE,
            data_points=full_dataset,
            targets=dataset.targets,
            spikes=dataset.spikes_list,
            sup_num=N_SUPERVISED,
            transform=composed,
        )

        # how often would a supervised batch be encountered during inference
        # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
        # until we have traversed through the all supervised batches
        periodic_interval_batches = (
            int(datasets.PyroCerebellumDataset.train_data_size / (1.0 * N_SUPERVISED))
            if N_SUPERVISED is not None
            else 0
        )

        # number of unsupervised examples
        unsup_num = (
            (datasets.PyroCerebellumDataset.train_data_size - N_SUPERVISED)
            if N_SUPERVISED is not None
            else datasets.PyroCerebellumDataset.train_data_size
        )

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc = 0.0

        # run inference for a certain number of epochs
        classifier_losses = []
        encoder_losses = []

        for i in tqdm(range(0, EPOCHS), desc="epoch", leave=False):

            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup = run_inference_for_epoch(
                data_loaders, losses, periodic_interval_batches, scheduler=scheduler
            )

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = map(lambda v: v / N_SUPERVISED, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = list(map(str, avg_epoch_losses_sup))
            str_loss_unsup = list(map(str, avg_epoch_losses_unsup))
            classifier_losses.append(str_loss_sup[0])
            encoder_losses.append(str_loss_unsup[0])

            str_print = f"{i} epoch: Sup loss: {str_loss_sup[0]}, Unsup loss: {str_loss_unsup[0]}"

            validation_accuracy = get_accuracy(
                data_loaders["valid"], ss_vae.classifier, BATCH_SIZE
            )

            train_accuracy = get_accuracy(
                data_loaders["sup"], ss_vae.classifier, BATCH_SIZE
            )

            str_print += f" - valid acc: {validation_accuracy}"
            str_print += f" - train: {train_accuracy}"

            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                # corresponding_test_acc = test_accuracy

            print_and_log(logger, str_print)

        print_and_log(
            logger,
            "best validation accuracy {} ".format(best_valid_acc),
        )
    except ValueError:
        raise optuna.exceptions.TrialPruned()

    finally:
        # close the logger file object if we opened it earlier
        if LOG:
            logger.close()

    return validation_accuracy, train_accuracy


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "SSVAE_v3"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        directions=["maximize", "maximize"],
    )

    study.optimize(objective, n_trials=300, timeout=None)

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
