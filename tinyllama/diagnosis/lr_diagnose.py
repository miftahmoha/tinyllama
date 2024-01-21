from copy import deepcopy
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..training import TrainConfig, Trainer
from ..diagnosis import Diagnose
from ..models import Llama


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

        TRAIN_CONFIG.__setattr__("epochs", self.epochs_for_each)
        Trainer_ = Trainer(TRAIN_CONFIG)

        for lr in tqdm(lrs, total=self.n_lrs):
            model_clone = deepcopy(model)

            optimizer = torch.optim.Adam(model_clone.parameters())

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            out = Trainer_.run(model, tokens, hide_progress=True)

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

        # plt.ylim(0, 5)
        plt.show()
