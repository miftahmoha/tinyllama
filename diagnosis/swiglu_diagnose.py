from typing import Dict, Optional, Callable
from copy import deepcopy
import json

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from training import train
from config import train_config, swiglu_config

# a dict to store the activations
activation = {}


def getActivation(name):
    # hook signature
    def hook(model, input, output):
        activation[name] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    return hook


def swiglu_diagnose(
    model,
    tokens: Tensor,
    context_window,
    num_embeddings_for_histogram: int = swiglu_config["num_embeddings_for_histogram"],
    track_direction: str = swiglu_config["track_direction"],
):

    model_clone = deepcopy(model)
    legends = []

    # hooking activations layers
    if track_direction == "forward":
        hook_activ_swiglu_0 = model_clone.llama_block_seq.llama_0.linear[
            1
        ].register_forward_hook(getActivation("SwiGLU activations 0"))
        hook_activ_swiglu_1 = model_clone.llama_block_seq.llama_1.linear[
            1
        ].register_forward_hook(getActivation("SwiGLU activations 1"))
        hook_activ_swiglu_2 = model_clone.linear[1].register_forward_hook(
            getActivation("SwiGLU activations 2")
        )

    elif track_direction == "backward":
        hook_activ_swiglu_0 = model_clone.llama_block_seq.llama_0.linear[
            1
        ].register_full_backward_hook(getActivation("SwiGLU activations 0"))
        hook_activ_swiglu_1 = model_clone.llama_block_seq.llama_1.linear[
            1
        ].register_full_backward_hook(getActivation("SwiGLU activations 1"))
        hook_activ_swiglu_2 = model_clone.linear[1].register_full_backward_hook(
            getActivation("SwiGLU activations 2")
        )
    else:
        raise ValueError("Parameter 'mode' can only be forward or backward!")

    # train the model
    optimizer = torch.optim.Adam(model_clone.parameters())
    train(model_clone, tokens, context_window, *train_config.values(), optimizer)

    # histograms
    for i in range(num_embeddings_for_histogram):
        hy, hx = torch.histogram(
            activation["SwiGLU activations 0"][0, i, :].cpu(), density=True
        )
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append("SwiGLU activations 0")
    hook_activ_swiglu_0.remove()
    plt.title("Histogram for the 1st SwiGLU layer (first batch):")
    plt.show()

    for i in range(num_embeddings_for_histogram):
        hy, hx = torch.histogram(
            activation["SwiGLU activations 1"][0, i, :].cpu(), density=True
        )
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append("SwiGLU activations 1")
    hook_activ_swiglu_1.remove()
    plt.title("Histogram for the 2nd SwiGLU layer (first batch)")
    plt.show()

    for i in range(num_embeddings_for_histogram):
        hy, hx = torch.histogram(
            activation["SwiGLU activations 2"][0, i, :].cpu(), density=True
        )
        plt.plot(hx[:-1].detach(), hy.detach())
    hook_activ_swiglu_2.remove()
    plt.title("Histogram for the 3rd SwiGLU layer (first batch):")
    plt.show()
