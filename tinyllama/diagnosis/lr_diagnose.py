from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..diagnosis import Diagnose
from ..models import Llama
from ..training import TrainConfig, Trainer


class LrDiagnose(Diagnose):
    def __init__(self, *, start: int, end: int, n_lrs: int):
        self.start = start
        self.end = end
        self.n_lrs = n_lrs
        self.epochs_for_each = 1

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):
        losses = []
        legends = []

        # create the tensor
        lrs = 10 ** np.linspace(self.start, self.end, self.n_lrs)

        TRAIN_CONFIG_copy = deepcopy(TRAIN_CONFIG)
        TRAIN_CONFIG_copy.__setattr__("epochs", self.epochs_for_each)

        model_clone = deepcopy(model)

        for lr in tqdm(lrs, total=self.n_lrs):
            TRAIN_CONFIG_copy["lr"] = lr
            Trainer_copy = Trainer(TRAIN_CONFIG_copy)

            out = Trainer_copy.run(model_clone, tokens, hide_progress=True)
            losses += [out]

        loss_train = [item[0]["train"] for item in losses]
        loss_val = [item[0]["val"] for item in losses]

        # plot loss_train and loss_val
        plt.plot(np.linspace(self.start, self.end, self.n_lrs), loss_train)
        legends.append("Training batches")

        plt.plot(np.linspace(self.start, self.end, self.n_lrs), loss_val)
        legends.append("Test batches")

        # add labels and a legend
        plt.xlabel("x")
        plt.ylabel("Loss")

        plt.legend(legends)

        plt.show()
