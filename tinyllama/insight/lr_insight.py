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

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):
        losses = []
        legends = []

        # create the tensor
        lrs = 10 ** np.linspace(self.start, self.end, self.n_lrs)

        TRAIN_CONFIG_copy = TRAIN_CONFIG.clone()
        TRAIN_CONFIG_copy.__setattr__("epochs", self.epochs_for_each)

        model_clone = model.clone()

        for lr in tqdm(lrs, total=self.n_lrs):
            TRAIN_CONFIG_copy["lr"] = lr
            Trainer_ = Trainer(TRAIN_CONFIG_copy)
            # [TODO] cache `DISABLE_TQDM`, then disable run
            out = Trainer_.run(model_clone, tokens)
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
