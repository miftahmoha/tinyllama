from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import itertools
import json

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from training import train
from config import train_config, gdratio_config


def gdratio_diagnose(
    model,
    tokens: Tensor,
    num_iters: int = gdratio_config["num_iters"],
    num_params_to_track: int = gdratio_config["num_params_to_track"],
):
    legends = []

    model_clone = deepcopy(model)

    gd_records = defaultdict(list)

    optimizer = torch.optim.Adam(model_clone.parameters())
    train_config.update({"epochs": 1})

    for _ in tqdm(range(num_iters)):
        train(model_clone, tokens, train_config, optimizer, show_progress=False)

        for name, param in itertools.islice(
            model_clone.named_parameters(), num_params_to_track
        ):
            if param.grad is not None:
                gd_records[name].append(
                    param.grad.cpu().std() / param.detach().cpu().std()
                )

    for name, gdr_value in gd_records.items():
        name = ".".join(name.split(".")[-2:])
        legends.append(f"Param name: {name}")
        plt.plot(gdr_value)

    plt.title("Gradient / Data ratio")
    plt.legend(legends)
    plt.show()
