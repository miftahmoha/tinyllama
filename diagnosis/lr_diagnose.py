from typing import Dict, Callable, Optional
from copy import deepcopy
import json
import tqdm

import torch
from torch import Tensor
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np

from training import train
from config import train_config, lr_config

"""
# reading config file for lr_diagnosis
json_file_path = "llama_config.json"
with open(json_file_path, "r") as json_file:
    MASTER_CONFIG = json.load(json_file)
"""

# train_config = MASTER_CONFIG["model"]
# lr_config = MASTER_CONFIG["lr_diagnosis"]


def lr_diagnose(model, tokens: Tensor):
    model_clone = deepcopy(model)

    losses = []
    legends = []

    # specify the range and the number of values
    start = lr_config["start"]
    end = lr_config["end"]

    # create the tensor
    lrs = 10 ** np.linspace(start, end, lr_config["n_lrs"])

    train_config["epochs"] = lr_config["epochs_for_each"]

    optimizer = torch.optim.Adam(model_clone.parameters())
    for lr in lrs:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        out = train(model_clone, tokens, train_config, optimizer, show_progress=False)
        losses += [out]

    loss_train = [item[0]["train"] for item in losses]
    loss_val = [item[0]["val"] for item in losses]

    # plot loss_train and loss_val on the same plot
    plt.plot(np.linspace(start, end, lr_config["n_lrs"]), loss_train)
    legends.append("Training batches")
    plt.plot(np.linspace(start, end, lr_config["n_lrs"]), loss_val)
    legends.append("Test batches")

    # add labels and a legend
    plt.xlabel("x")
    plt.ylabel("Loss")
    # plt.ylim(0, 5)
    plt.show()
