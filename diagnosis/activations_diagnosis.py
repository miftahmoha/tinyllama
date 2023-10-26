from typing import Dict, Optional, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
import matplotlib.pyplot as plt

from models import Llama

# a dict to store the activations
activation = {}


def getActivation(name):
    # hook signature
    def hook(model, input, output):
        activation[name] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    return hook


def activations_diagnosis_wrapper(
    train: Callable[
        [Llama, str, Dict, Optimizer, Optional[bool], Optional[bool]], Tensor
    ],
    mode: str = "forward",
):
    def wrapper(*args, **kwargs):
        legends = []

        # hooking activations layers
        if mode == "forward":
            hook_activ_swiglu_0 = (
                args[0]
                .llama_block_seq.llama_0.linear[1]
                .register_forward_hook(getActivation("SwiGLU activations 0"))
            )
            hook_activ_swiglu_1 = (
                args[0]
                .llama_block_seq.llama_1.linear[1]
                .register_forward_hook(getActivation("SwiGLU activations 1"))
            )
            hook_activ_swiglu_2 = (
                args[0]
                .linear[1]
                .register_forward_hook(getActivation("SwiGLU activations 2"))
            )
        elif mode == "backward":
            hook_activ_swiglu_0 = (
                args[0]
                .llama_block_seq.llama_0.linear[1]
                .register_full_backward_hook(getActivation("SwiGLU activations 0"))
            )
            hook_activ_swiglu_1 = (
                args[0]
                .llama_block_seq.llama_1.linear[1]
                .register_full_backward_hook(getActivation("SwiGLU activations 1"))
            )
            hook_activ_swiglu_2 = (
                args[0]
                .linear[1]
                .register_full_backward_hook(getActivation("SwiGLU activations 2"))
            )
        else:
            raise ValueError("Parameter 'mode' can only be forward or backward!")

        print(f"Hooking {args[0].linear[1]}..")
        train(*args, **kwargs)
        print("Return activations for: {args[0].linear[1]}..")

        # histograms
        print("Histogram 0 for the first batch:")
        for i in range(5):
            hy, hx = torch.histogram(
                activation["SwiGLU activations 0"][0, i, :].cpu(), density=True
            )
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append("SwiGLU activations 0")
        hook_activ_swiglu_0.remove()
        plt.title("SwiGLU activations 0")
        plt.show()

        print("Histogram 1 for the first batch:")
        for i in range(5):
            hy, hx = torch.histogram(
                activation["SwiGLU activations 1"][0, i, :].cpu(), density=True
            )
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append("SwiGLU activations 1")
        hook_activ_swiglu_1.remove()
        plt.title("SwiGLU activations 1")
        plt.show()

        print("Histogram 2 for the first batch:")
        for i in range(5):
            hy, hx = torch.histogram(
                activation["SwiGLU activations 2"][0, i, :].cpu(), density=True
            )
            plt.plot(hx[:-1].detach(), hy.detach())
        hook_activ_swiglu_2.remove()
        plt.title("SwiGLU activations 2")
        plt.show()

    return wrapper
