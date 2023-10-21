from typing import Dict, Callable, Optional

from torch import Tensor
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import numpy as np

from models import Llama


def lrs_diagnosis_wrapper(
    train: Callable[
        [Llama, str, Dict, Optimizer, Optional[bool], Optional[bool]], Tensor
    ],
    n_lrs=32,
):
    def wrapper(
        *args,
        **kwargs,
    ):
        losses = []
        legends = []

        # specify the range and the number of values
        start = -5
        end = 0

        # create the tensor
        lrs = 10 ** np.linspace(start, end, n_lrs)

        args[2]["epochs"] = 5
        print(f'The number of epochs for each lr is set to {args[2]["epochs"]}')

        for lr in lrs:
            for param_group in args[3].param_groups:
                param_group["lr"] = lr
            out = train(*args, **kwargs)
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
        plt.ylim(0, 5)
        plt.show()

    return wrapper
