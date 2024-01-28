"""
Hyperparameter tuner for tinyllama using Bayesian implementation of a noiseless Gaussian Process using STAN.
"""

import os
from copy import deepcopy

import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
import stan
import torch
from scipy.stats import qmc
from tqdm import tqdm

from ..diagnosis import Diagnose
from ..models import Llama
from ..training import TrainConfig, Trainer


# reads the STAN model
script_directory = os.path.dirname(os.path.realpath(__file__))
with open(script_directory + "/gptuner.stan", "r") as file:
    gptuner_stan = file.read()


def process_results(
    results: stan.fit.Fit,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    N_val: int,
    hyperparam_to_tune: list[str],
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
    :param N_val: Number of evaluation samples
    :type N_val: Int
    """

    df_results = results.to_frame().describe().T
    Y_test_mean = df_results["Y_test.1" : "Y_test." + str(N_val)]["mean"].values
    Y_test_25qtl = df_results["Y_test.1" : "Y_test." + str(N_val)]["25%"].values
    Y_test_75qtl = df_results["Y_test.1" : "Y_test." + str(N_val)]["75%"].values

    # Processing
    X = np.vstack((X_train, X_test))
    Y_mean = np.hstack((Y_train, Y_test_mean))
    Y_25qtl = np.hstack((Y_train, Y_test_25qtl))
    Y_75qtl = np.hstack((Y_train, Y_test_75qtl))

    # Concatenating
    combined_matrix = np.column_stack((X, Y_25qtl, Y_mean, Y_75qtl))

    dataframe = pd.DataFrame(
        combined_matrix, columns=hyperparam_to_tune + ["Y_25qtl", "Y_mean", "Y_75qtl"]
    )

    """
    # Create 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # needs changes to be able to make smooth plots
    ax.plot_trisurf(X[:, 0], X[:, 1], Y_mean, color="grey")
    ax.plot_trisurf(X[:, 0], X[:, 1], Y_25qtl, color="green")
    ax.plot_trisurf(X[:, 0], X[:, 1], Y_75qtl, color="red")

    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c="red", marker="o")

    ax.set_xlabel("Epoches")
    ax.set_ylabel("Embedding Dimension")
    ax.set_zlabel("Loss")
    ax.set_title("3D Surface Plot")

    plt.show()
    """

    return dataframe


class GPTuneConfig:
    num_training_samples: int
    l_bounds: list[int]
    u_bounds: list[int]
    hyperparams_to_tune: list[str]
    num_evaluations: int

    def __init__(self, **kwargs):
        try:
            self.num_training_samples = kwargs.pop("num_training_samples")
            self.l_bounds = kwargs.pop("l_bounds")
            self.u_bounds = kwargs.pop("u_bounds")
            self.hyperparams_to_tune = kwargs.pop("hyperparams_to_tune")
            self.num_evaluations = kwargs.pop("num_evaluations")
        except KeyError as e:
            print(f"Missing keyword argument {e}=...in GPTuneConfig")

    def __getitem__(self, name: str):
        return self.__getattribute__(name)

    def __setitem__(self, name: str, value: int):
        self.__setattr__(name, value)


class GPTune(Diagnose):
    def __init__(self, GPTUNE_CONFIG: GPTuneConfig):
        self.GPTUNE_CONFIG = GPTUNE_CONFIG

    def run(
        self,
        model: Llama,
        tokens: torch.Tensor,
        TRAIN_CONFIG: TrainConfig,
        num_stan_samples: int = 50,
    ):
        N_train, M = (
            self.GPTUNE_CONFIG["num_training_samples"],
            len(self.GPTUNE_CONFIG["hyperparams_to_tune"]),
        )

        # get latin hypercube distributed samples for hyperparameters
        sampler = qmc.LatinHypercube(d=M)
        sample = sampler.random(n=N_train)

        X_train = qmc.scale(
            sample, self.GPTUNE_CONFIG["l_bounds"], self.GPTUNE_CONFIG["u_bounds"]
        )

        # training & retrieve validation error for each sampled hyperparameter
        Y_train = np.array([])
        for hyperparam in tqdm(X_train, total=N_train, colour="blue"):
            model_clone = deepcopy(model)

            for index, hyperparam_to_tune in enumerate(
                self.GPTUNE_CONFIG["hyperparams_to_tune"]
            ):
                if hyperparam_to_tune in [
                    "context_window",
                    "emb_dim",
                    "n_heads",
                    "n_blocks",
                ]:
                    setattr(model_clone, hyperparam_to_tune, round(hyperparam[index]))

                elif hyperparam_to_tune in ["epochs", "batch_size", "log_size"]:
                    TRAIN_CONFIG[hyperparam_to_tune] = round(hyperparam[index])

                else:
                    raise ValueError(
                        f"The parameter {hyperparam_to_tune} is inexistant"
                    )

            Y_train = np.append(
                Y_train,
                float(
                    Trainer(TRAIN_CONFIG).run(model_clone, tokens, hide_progress=True)[
                        -1
                    ]["train"]
                ),
            )

        # N_train, M = X_train.shape

        # generating test samples for hyperparameters (going with uniform but could be abstracted)
        N_val = self.GPTUNE_CONFIG["num_evaluations"]
        X_test = np.random.uniform(
            low=self.GPTUNE_CONFIG["l_bounds"],
            high=self.GPTUNE_CONFIG["u_bounds"],
            size=(N_val, M),
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
        fit_results = posterior.sample(num_chains=4, num_samples=num_stan_samples)
        # print(type(posterior), type(fit_results), type(X_train), type(Y_train), type(X_test))
        results = process_results(
            fit_results,
            X_train,
            Y_train,
            X_test,
            N_val,
            self.GPTUNE_CONFIG["hyperparams_to_tune"],
        )

        return results
