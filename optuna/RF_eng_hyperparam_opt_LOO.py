import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

import optuna
from optuna.trial import TrialState


from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

NRUNS = 5
SEED = 1234

# get relative data folder
PATH = os.path.dirname(os.path.abspath(""))
data_folder = PATH + "/celltypes-classification/data"
DATA_PATH = data_folder + ("/Aug-09-2022_all_features.csv")


df = pd.read_csv(DATA_PATH, index_col=0)


def filter_df(df: pd.DataFrame):
    """
    Filters out datapoints with unusable temporal features.
    """
    features_only = df.iloc[:, 2:]
    bad_idx = []
    for i, row in features_only.iterrows():
        value, count = np.unique(row.to_numpy(), return_counts=True)
        zeros = count[value == 0]
        if zeros.size > 0 and zeros > 5:
            bad_idx.append(i)
    keep = [i for i in range(len(df)) if i not in bad_idx]
    return df.iloc[keep]


def generate_train_and_labels(df: pd.DataFrame, info_idx=[0, 1, 2, 18]):
    info = df.iloc[:, info_idx]
    features = df.iloc[:, ~np.isin(np.arange(len(df.columns)), info_idx)]
    return features.copy(), info.iloc[:, 0].copy()


X, y = generate_train_and_labels(filter_df(df))


def define_model(trial: optuna.trial.Trial):

    n_estimators = trial.suggest_int("n_estimators", 100, 500)

    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    oob_score = trial.suggest_categorical("oob_score", [True, False])

    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=oob_score,
    )


def objective(trial):

    # Generate the model.
    kfold = LeaveOneOut()

    true_targets = []
    model_pred = []
    f1_scores = []

    for run in range(NRUNS):
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(X, y)),
            leave=True,
            position=0,
            desc="Cross-validating",
            total=len(X),
        ):

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[val_idx]
            y_test = y.iloc[val_idx]

            oversample = RandomOverSampler(random_state=SEED)

            X_big, y_big = oversample.fit_resample(X_train, y_train)
            model = define_model(trial)

            # fit the model on the data
            model.fit(X_big, y_big)
            pred = model.predict(X_test)

            true_targets.append(y_test)
            model_pred.append(pred)

        f1 = f1_score(true_targets, model_pred, average="macro")
        f1_scores.append(f1)
        trial.report(f1, run)

    return np.array(f1_scores).mean()


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "random-forest-feat-eng_LOO"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=200, timeout=None)

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
