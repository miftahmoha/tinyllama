import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from tinyllama.insight import Insight
from tinyllama.models import Llama
from tinyllama.training import TrainConfig, Trainer


class LrInsight(Insight):
    def __init__(self, *, start: int, end: int, n_lrs: int):
        self.start = start
        self.end = end
        self.n_lrs = n_lrs
        self.epochs_for_each = 1

    def run(
        self,
        model: Llama,
        tokens: torch.Tensor,
        TUNE_CONFIG: TrainConfig = TrainConfig(batch_size=32, epochs=64),
        tune_on_clone: bool = True,
    ):
        losses = []
        legends = []

        # create the tensor
        lrs = 10 ** np.linspace(self.start, self.end, self.n_lrs)

        TUNE_CONFIG_ = TUNE_CONFIG.clone()
        TUNE_CONFIG_.__setattr__("epochs", self.epochs_for_each)

        model_ = model.clone() if tune_on_clone else model

        for lr in tqdm(lrs, total=self.n_lrs):
            TUNE_CONFIG_["lr"] = lr
            Trainer_ = Trainer(TUNE_CONFIG_)
            out = Trainer_.run(model_, tokens)
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
