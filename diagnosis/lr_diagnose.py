from typing import Dict, Callable, Optional
from copy import deepcopy
import json
import tqdm

import torch
from torch import nn
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np

from training import train
from config import train_config, lr_config


def lr_diagnose(
    model: nn.Module,
    tokens: Tensor,
    context_window: int,
    start: int = lr_config["start"],
    end: int = lr_config["end"],
    epochs_for_each: int = lr_config["epochs_for_each"],
    n_lrs: int = lr_config["n_lrs"],
):
    model_clone = deepcopy(model)

    losses = []
    legends = []

    # create the tensor
    lrs = 10 ** np.linspace(start, end, n_lrs)

    # train_config["epochs"] = epochs_for_each
    train_config.update({"epochs": epochs_for_each})

    optimizer = torch.optim.Adam(model_clone.parameters())
    for lr in lrs:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # CAUTION: the order of the args in the json file will cause bugs if changed!
        out = train(
            model_clone,
            tokens,
            context_window,
            *train_config.values(),
            optimizer,
            show_progress=False
        )
        losses += [out]

    loss_train = [item[0]["train"] for item in losses]
    loss_val = [item[0]["val"] for item in losses]

    # plot loss_train and loss_val on the same plot
    plt.plot(np.linspace(start, end, n_lrs), loss_train)
    legends.append("Training batches")
    plt.plot(np.linspace(start, end, n_lrs), loss_val)
    legends.append("Test batches")

    # add labels and a legend
    plt.xlabel("x")
    plt.ylabel("Loss")
    # plt.ylim(0, 5)
    plt.show()
