from typing import Dict, Optional, Callable
import itertools

import torch
from torch import Tensor
from torch.optim import Optimizer
import matplotlib.pyplot as plt

from models import Llama


def gradients_diagnosis_wrapper(
    train: Callable[
        [Llama, str, Dict, Optimizer, Optional[bool], Optional[bool]], Tensor
    ]
):
    def wrapper(*args, **kwargs):
        legends = []
        train(*args, **kwargs)
        for name, param in itertools.islice(args[0].named_parameters(), 30):
            if param.grad is not None:
                # Access the gradients for the parameter
                gradients = param.grad
                hy, hx = torch.histogram(gradients.cpu(), density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                name = ".".join(name.split(".")[-2:])
                legends.append(f"Param name: {name}")
        plt.title("Gradient density.")
        plt.legend(legends)
        plt.show()

    return wrapper
