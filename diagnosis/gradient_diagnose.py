from typing import Dict, Optional, Callable
from copy import deepcopy
import itertools
import json

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from training import train
from config import train_config, gradient_config


def gradient_diagnose(
    model,
    tokens: Tensor,
    num_params_to_track: int = gradient_config["num_params_to_track"],
):
    legends = []

    model_clone = deepcopy(model)

    optimizer = torch.optim.Adam(model_clone.parameters())
    train(model_clone, tokens, train_config, optimizer)

    for name, param in itertools.islice(
        model_clone.named_parameters(), num_params_to_track
    ):
        if param.grad is not None:
            # Access the gradients for the parameter
            gradients = param.grad
            hy, hx = torch.histogram(gradients.cpu(), density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            name = ".".join(name.split(".")[-2:])
            legends.append(f"Param name: {name}")

    plt.title("Gradient density")
    plt.legend(legends)
    plt.show()
