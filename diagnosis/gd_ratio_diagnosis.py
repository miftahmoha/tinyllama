from typing import Dict, Optional, Callable
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import Tensor
from torch.optim import Optimizer
import matplotlib.pyplot as plt

from models import Llama


def gd_diagnosis_wrapper(
    train: Callable[
        [Llama, str, Dict, Optimizer, Optional[bool], Optional[bool]], Tensor
    ]
):
    def wrapper(*args, **kwargs):
        gd_records = defaultdict(list)
        n_iters = 1000
        args[2]["epochs"] = 1

        for _ in tqdm(range(n_iters)):
            train(*args, **kwargs, show_progress=False)
            for name, param in args[0].named_parameters():
                if param.grad is not None:
                    gd_records[name].append(
                        param.grad.cpu().std() / param.detach().cpu().std()
                    )

        for value in gd_records.values():
            plt.plot(value)

        plt.show()

    return wrapper
