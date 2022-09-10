from pathlib import Path
import torch

SEED = 1234
# Path to the current hdf5 dataset
home = str(Path.home())
DATA_PATH = home + "/cerebellum_dataset.h5"

# To convert text labels to numbers
LABELLING = {
    "PkC_cs": 5,
    "PkC_ss": 4,
    "MFB": 3,
    "MLI": 2,
    "GoC": 1,
    "GrC": 0,
    "unlabelled": -1,
}

# To do the inverse
CORRESPONDENCE = {
    5: "PkC_cs",
    4: "PkC_ss",
    3: "MFB",
    2: "MLI",
    1: "GoC",
    0: "GrC",
    -1: "unlabelled",
}

CENTRAL_RANGE = 60

N_CHANNELS = 10

TEST_SIZE = 0.2

BATCH_SIZE = 50

ACG_LEN = 100

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACG_SCALING = 331.3285579132315

## Pyro related: ##

TRAIN_SIZE = 954

TEST_SIZE = 0

VALIDATION_SIZE = 0.1

N_CLASSES = 6

## End of pyro related ##
