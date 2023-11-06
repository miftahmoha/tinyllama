"""
Hyperparameter tuner for tinyllama using Bayesian implementation of a noiseless Gaussian Process using STAN.
"""
from typing import Callable, Dict

from torch.optim import Optimizer
import numpy as np
import stan
from scipy.stats import qmc
import matplotlib.pyplot as plt

from models import Llama

# reads the STAN model
with open("./gptuner/gptuner.stan", "r") as file:
    gptuner_stan = file.read()


def process_results(results, X_train, Y_train, X_test, N_val):
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

    # Create 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(X[:, 0], X[:, 1], Y_mean, color="grey")
    ax.plot_trisurf(X[:, 0], X[:, 1], Y_25qtl, color="green")
    ax.plot_trisurf(X[:, 0], X[:, 1], Y_75qtl, color="red")

    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c="red", marker="o")

    ax.set_xlabel("Epoches")
    ax.set_ylabel("Embedding Dimension")
    ax.set_zlabel("Loss")
    ax.set_title("3D Surface Plot")

    plt.show()


def gptune(
    train: Callable,
    model: Llama,
    tokens: Dict,
    optimizer: Optimizer,
    MASTER_CONFIG: Dict,
):
    """
    Performs hyperparameter tuning using Gaussian Processes

    .. comment ::

        We'll start by implementing a noiseless GP and only tune two parameters, we'll add more abstractions
        to generalize over other hyperparameters and GPs such as Noisy GPs.

    :param model: Llama model
    :type model: Callable
    :param train: Training utility for Llama
    :type train: Callable
    :param MASTER_CONFIG: Dictionary containing the hyperparameters
    :type MASTER_CONFIG: Dict
    """

    # get latin hypercube distributed samples for hyperparameters
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=50)

    l_bounds = [25, 2]
    u_bounds = [35, 20]
    X_train = qmc.scale(sample, l_bounds, u_bounds)

    # training & retrieve validation error for each sampled hyperparameter
    Y_train = []
    for hyperparam in X_train:
        MASTER_CONFIG["epochs"] = int(hyperparam[0])
        MASTER_CONFIG["emb_dim"] = hyperparam[1]
        Y_train += [float(train(model, tokens, MASTER_CONFIG, optimizer)[-1]["val"])]
    N_train, M = X_train.shape

    # generating test samples for hyperparameters (going with uniform but could be abstracted)
    N_val = 50
    X_test = np.random.uniform(low=l_bounds, high=u_bounds, size=(N_val, M))

    data = {
        "N_train": N_train,
        "N_val": N_val,
        "M": M,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
    }
    posterior = stan.build(gptuner_stan, data=data)
    fit_results = posterior.sample(num_chains=4, num_samples=100)
    results = process_results(fit_results, X_train, Y_train, X_test, N_val)
    return results
