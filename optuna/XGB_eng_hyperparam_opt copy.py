import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import sys

import optuna
from optuna.trial import TrialState


from npyx.feat import filter_df
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix

import xgboost as xgb

seed = np.random.seed(123)

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

LABELLING = {"PkC_cs": 5, "PkC_ss": 4, "MFB": 3, "MLI": 2, "GoC": 1, "GrC": 0}
y.replace(to_replace=LABELLING, inplace=True)


def define_model(trial: optuna.trial.Trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.

    n_estimators = trial.suggest_int("n_estimators", 10, 100)

    booster = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])

    max_depth = trial.suggest_categorical("max_depth", [0, 4, 5, 6, 7, 8, 9, 10])

    learning_rate = trial.suggest_float("learning_rate", 0.0, 1.0, step=0.01)

    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 5.0, step=0.1)

    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0, step=0.1)

    subsample = trial.suggest_float("subsample", 0.1, 1.0, step=0.1)

    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        booster=booster,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )


def objective(trial: optuna.trial.Trial):

    # Generate the model.
    f1_scores = []

    model = define_model(trial)

    kfold = StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in tqdm(
        enumerate(kfold.split(X, y)),
        leave=True,
        position=0,
        desc="Cross-validating",
        total=5,
    ):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[val_idx]
        y_test = y.iloc[val_idx]

        oversample = RandomOverSampler(random_state=seed)

        X_big, y_big = oversample.fit_resample(X_train, y_train)

        # D_train = xgb.DMatrix(X_big, label=y_big)
        # D_test = xgb.DMatrix(X_test, y_test)

        model.fit(X_big, y_big)

        pred = model.predict(X_test)

        fold_f1 = f1_score(y_test, pred, average="weighted")
        f1_scores.append(fold_f1)
        trial.report(np.array(f1_scores).mean(), fold)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.array(f1_scores).mean()


def main():

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "XGB-feat-eng"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(objective, n_trials=50, timeout=None)

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
