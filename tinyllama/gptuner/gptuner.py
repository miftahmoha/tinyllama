"""
Hyperparameter tuner for tinyllama using Bayesian implementation of a noiseless Gaussian Process using STAN.
"""

import os
import sys
import warnings
from itertools import islice, product
from typing import TextIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stan
import torch
from scipy import interpolate, stats
from tqdm import tqdm

from tinyllama.globals import DISABLE_PLOT, QUIET_STAN
from tinyllama.insight import Insight
from tinyllama.models import Llama
from tinyllama.training import TrainConfig, Trainer


def generate_combination_matrix(arrays, num_combinations):
    # generate all possible combinations of elements from the arrays
    combinations = product(*arrays)

    unique_combinations = set()

    for combo in combinations:
        if len(set(combo)) == len(combo):  # check for uniqueness
            unique_combinations.add(combo)

    matrix = [list(combo) for combo in unique_combinations]

    matrix = list(islice(matrix, num_combinations))

    return matrix, len(matrix)


def generate_integer_samples(lower_bound, upper_bound, max_num_eval_samples):
    M = len(upper_bound)

    int_params = []
    for i in range(M):
        int_param = np.arange(lower_bound[i], upper_bound[i] + 1)
        int_params += [int_param]

    int_samples, num_eval_samples = generate_combination_matrix(
        int_params, max_num_eval_samples
    )

    return np.array(int_samples), num_eval_samples


def make_training_set_unique(X_train: np.ndarray | list | set):
    X_train = [list(item) for item in {tuple(item) for item in X_train}]


# redirects output streams
def redirect_streams(stdout: TextIO, stderr: TextIO):
    streams = (sys.stdout, sys.stderr)
    sys.stdout = stdout
    sys.stderr = stderr
    return streams


class QuietStan:
    def __enter__(self):
        if QUIET_STAN:
            self.dev_null = open(os.devnull, "w")
            self.stdout, self.stderr = redirect_streams(self.dev_null, self.dev_null)

    def __exit__(self, exc_type, exc_value, traceback):
        if QUIET_STAN:
            _, _ = redirect_streams(self.stdout, self.stderr)
            self.dev_null.close()


# reads the STAN model
script_directory = os.path.dirname(os.path.realpath(__file__))
with open(script_directory + "/gptuner.stan") as file:
    gptuner_stan = file.read()


def process_results(
    results: stan.fit.Fit,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    N_val: int,
    hyperparams_to_tune: list[str],
    hyperparams_to_plot: list[str],
):
    """
    Process results and plots a 3D plot with the mean, 25% and 75% percentiles.

    :param results: Stan output result
    :type model: Fit object
    :param X_train: Training samples for noiseless GP
    :type X_train: List
    :param Y_train: Loss samples for noiseless GP
    :type X_train: List
    :param X_test: Test samples for noiseless GP
    :type X_test: List
    """

    # retrieving results
    df_results = results.to_frame().describe().T
    Y_test_25qt = df_results.loc["Y_test.1" : "Y_test." + str(N_val)]["25%"].values  # type: ignore
    Y_test_mean = df_results.loc["Y_test.1" : "Y_test." + str(N_val)]["mean"].values  # type: ignore
    Y_test_75qt = df_results.loc["Y_test.1" : "Y_test." + str(N_val)]["75%"].values  # type: ignore

    # stacking matrices for X_train & X_test
    train_matrix = np.column_stack((X_train, Y_train))
    eval_matrix = np.column_stack((X_test, Y_test_25qt, Y_test_mean, Y_test_75qt))

    # creating datafranes for X_train & X_test
    train_df = pd.DataFrame(train_matrix, columns=hyperparams_to_tune + ["Y_train"])
    eval_df = pd.DataFrame(
        eval_matrix, columns=hyperparams_to_tune + ["Y_25qtl", "Y_mean", "Y_75qtl"]
    )

    if DISABLE_PLOT:
        warnings.warn(
            "`DISABLE_PLOT` is set to 0, processing without plotting..", UserWarning
        )
    else:
        if X_train.shape[1] == 1 and not hyperparams_to_plot:
            plot_results_2D(
                X_train,
                Y_train,
                X_test,
                Y_test_25qt,
                Y_test_mean,
                Y_test_75qt,
                hyperparams_to_tune,
            )

        elif X_train.shape[1] == 2 and not hyperparams_to_plot:
            plot_results_3D(
                X_train,
                Y_train,
                X_test,
                Y_test_25qt,
                Y_test_mean,
                Y_test_75qt,
                hyperparams_to_tune,
            )

        else:
            indices: list[int] = (
                [
                    hyperparams_to_tune.index(hyperparam_to_plot)
                    for hyperparam_to_plot in hyperparams_to_plot
                ]
                if hyperparams_to_plot
                else []
            )
            labels: list[str] = sorted(
                hyperparams_to_plot, key=lambda x: hyperparams_to_tune.index(x)
            )

            if len(indices) == 1:
                plot_results_2D(
                    X_train[:, indices[0]],
                    Y_train,
                    X_test[:, indices[0]],
                    Y_test_25qt,
                    Y_test_mean,
                    Y_test_75qt,
                    labels[0],
                )

            elif len(indices) == 2:
                plot_results_3D(
                    X_train[:, indices],
                    Y_train,
                    X_test[:, indices],
                    Y_test_25qt,
                    Y_test_mean,
                    Y_test_75qt,
                    labels,
                )

            else:
                warnings.warn(
                    "`hyperparams_to_plot` is not set, processing without plotting..",
                    UserWarning,
                )

    return train_df, eval_df


def plot_results_2D(
    X_train, Y_train, X_test, Y_test_25qt, Y_test_mean, Y_test_75qt, label
):
    # create the figure
    plt.figure(figsize=(10, 6))

    # order indices
    indices = np.argsort(X_test)

    # plot mean prediction
    plt.plot(
        X_test[indices],
        Y_test_25qt[indices],
        color="black",
        linewidth=2,
        label="25th Percentile",
    )
    plt.plot(
        X_test[indices], Y_test_mean[indices], color="red", linewidth=2, label="Mean"
    )
    plt.plot(
        X_test[indices],
        Y_test_75qt[indices],
        color="orange",
        linewidth=2,
        label="75th Percentile",
    )

    # plot training data points
    plt.scatter(
        X_train, Y_train, color="orange", marker="o", alpha=0.6, label="Training Data"
    )

    # plot uncertainty region
    plt.fill_between(
        X_test[indices],
        Y_test_25qt[indices],
        Y_test_75qt[indices],
        color="gray",
        alpha=0.2,
        label="Uncertainty Region",
    )

    plt.xlabel(label)
    plt.ylabel("Loss")
    plt.title("2D Surface Plot")
    plt.legend()

    plt.show()


def plot_results_3D(
    X_train, Y_train, X_test, Y_test_25qtl, Y_test_mean, Y_test_75qtl, labels
):
    # create grid for the surface
    grid_x, grid_y = np.mgrid[
        min(X_test[:, 0]) : max(X_test[:, 0]) : 100j,
        min(X_test[:, 1]) : max(X_test[:, 1]) : 100j,
    ]

    # interpolate the data onto the grid
    grid_z_25qt = interpolate.griddata(
        (X_test[:, 0], X_test[:, 1]), Y_test_25qtl, (grid_x, grid_y), method="cubic"
    )
    grid_z_mean = interpolate.griddata(
        (X_test[:, 0], X_test[:, 1]), Y_test_mean, (grid_x, grid_y), method="cubic"
    )
    grid_z_75qt = interpolate.griddata(
        (X_test[:, 0], X_test[:, 1]), Y_test_75qtl, (grid_x, grid_y), method="cubic"
    )

    # create 3d surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plot surfaces
    ax.plot_surface(
        grid_x, grid_y, grid_z_25qt, alpha=0.4, color="green", label="25th Percentile"
    )
    ax.plot_surface(grid_x, grid_y, grid_z_mean, alpha=0.6, color="red", label="Mean")
    ax.plot_surface(
        grid_x, grid_y, grid_z_75qt, alpha=0.4, color="yellow", label="75th Percentile"
    )

    # scatter plot of training data
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        Y_train,
        c="orange",
        marker="o",
        label="Training Data",
    )

    # [TODO] need to deal with `nan` values?
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("Loss")
    ax.set_title("3D Surface Plot")
    ax.legend()

    plt.show()


class GPTuneConfig:
    max_num_training_samples: int
    l_bounds: list[int]
    u_bounds: list[int]
    hyperparams_to_tune: list[str]
    hyperparams_to_plot: list[str]
    max_num_evaluation_samples: int

    def __init__(self, **kwargs):
        try:
            self.max_num_training_samples = kwargs.pop("max_num_training_samples")
            self.l_bounds = kwargs.pop("l_bounds")
            self.u_bounds = kwargs.pop("u_bounds")
            self.hyperparams_to_tune = kwargs.pop("hyperparams_to_tune")
            self.hyperparams_to_plot = kwargs.pop("hyperparams_to_plot", [])
            self.max_num_evaluation_samples = kwargs.pop("max_num_evaluation_samples")
        except KeyError as e:
            print(f"Missing keyword argument {e}=...in GPTuneConfig")
            raise SystemExit

    def __getitem__(self, name: str):
        return self.__getattribute__(name)

    def __setitem__(self, name: str, value: int):
        self.__setattr__(name, value)


class GPTune(Insight):
    def __init__(self, GPTUNE_CONFIG: GPTuneConfig):
        self.GPTUNE_CONFIG = GPTUNE_CONFIG

    def run(
        self,
        model: Llama,
        tokens: torch.Tensor,
        TUNE_CONFIG: TrainConfig = TrainConfig(batch_size=32, epochs=64),
        num_stan_samples: int = 50,
    ):
        N_train, M = (
            self.GPTUNE_CONFIG["max_num_training_samples"],
            len(self.GPTUNE_CONFIG["hyperparams_to_tune"]),
        )

        # get latin hypercube distributed samples for hyperparameters
        sampler = stats.qmc.LatinHypercube(d=M)
        sample = sampler.random(n=N_train)

        # samples should be integers
        X_train = stats.qmc.scale(
            sample, self.GPTUNE_CONFIG["l_bounds"], self.GPTUNE_CONFIG["u_bounds"]
        ).astype(int)

        # removing duplicates (causes covariance matrix to not be positive definite)
        make_training_set_unique(X_train)

        # training & retrieve validation error for each sampled hyperparameter
        Y_train = np.array([])
        for hyperparam in tqdm(X_train, total=N_train, colour="cyan"):
            model_settings: dict[str, int] = {}

            for index, hyperparam_to_tune in enumerate(
                self.GPTUNE_CONFIG["hyperparams_to_tune"]
            ):
                if hyperparam_to_tune in [
                    "epochs",
                    "batch_size",
                    "lr",
                    "context_window",
                    "log_size",
                ]:
                    TUNE_CONFIG[hyperparam_to_tune] = hyperparam[index]

                elif hyperparam_to_tune in [
                    "context_window",
                    "emb_dim",
                    "n_heads",
                    "n_blocks",
                    "gq_ratio",
                    "vocab_size",
                ]:
                    model_settings[hyperparam_to_tune] = hyperparam[index]

                else:
                    raise ValueError(
                        f"The parameter {hyperparam_to_tune} is inexistant"
                    )

            model_ = model.new(**model_settings) if model_settings else model.clone()
            Y_train = np.append(
                Y_train,
                float(Trainer(TUNE_CONFIG).run(model_, tokens)[-1]["train"]),
            )

        # generating test samples for hyperparameters
        N_val = self.GPTUNE_CONFIG["max_num_evaluation_samples"]

        X_test, N_val = generate_integer_samples(
            self.GPTUNE_CONFIG["l_bounds"], self.GPTUNE_CONFIG["u_bounds"], N_val
        )

        data = {
            "N_train": N_train,
            "N_val": N_val,
            "M": M,
            "X_train": X_train,
            "X_test": X_test,
            "Y_train": Y_train,
        }

        posterior = stan.build(gptuner_stan, data=data)

        # redirects output streams to devnull
        # dev_null = open(os.devnull, "w")
        # stdout, stderr = redirect_streams(dev_null, dev_null)
        with QuietStan():
            fit_results = posterior.sample(num_chains=4, num_samples=num_stan_samples)

        # redirects them back
        # redirect_streams(stdout, stderr)
        # dev_null.close()

        results = process_results(
            fit_results,
            X_train,
            Y_train,
            X_test,
            N_val,
            self.GPTUNE_CONFIG["hyperparams_to_tune"],
            self.GPTUNE_CONFIG["hyperparams_to_plot"],
        )

        return results
