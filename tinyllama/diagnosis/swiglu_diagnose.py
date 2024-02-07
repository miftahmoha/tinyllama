from copy import deepcopy
from enum import Enum

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..diagnosis import Diagnose
from ..models import Llama
from ..training import TrainConfig, Trainer


class SwigluPath(Enum):
    FORWARD = 0
    BACKWARD = 1


# a dict to store the activations
activation = {}


def getActivation(name):
    # hook signature
    def hook(model, input, output):
        activation[name] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    return hook


def register_layer(layer: nn.Module, name: str, track_direction: SwigluPath):
    hook = (
        layer.register_forward_hook(getActivation(name))
        if track_direction == SwigluPath.FORWARD
        else layer.register_full_backward_hook(getActivation(name))
    )
    return hook


def plot_layer(layer_name: str, color: str):
    hy, hx = torch.histogram(
        activation[layer_name][:, :, :].cpu(),
        density=True,
    )
    plt.plot(hx[:-1].detach(), hy.detach(), color=color)


def compute_saturation(layer_name: str, precision: float = 1e-4):
    num_close_to_zero = torch.sum(
        torch.abs(activation[layer_name][:, :, :].cpu()) < precision
    ).item()
    total_elements = torch.prod(
        torch.tensor(activation[layer_name][:, :, :].cpu().shape)
    ).item()
    return num_close_to_zero / total_elements


class SwigluDiagnose(Diagnose):
    def __init__(self, *, track_direction: SwigluPath):
        self.track_direction = track_direction

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):
        model_clone = deepcopy(model)

        # hooking activations layers
        if self.track_direction == SwigluPath.FORWARD:
            hook_activ_swiglu_0 = register_layer(
                model_clone.llama_block_seq.llama_0.linear[1],
                "SwiGLU: Layer 0",
                SwigluPath.FORWARD,
            )
            hook_activ_swiglu_1 = register_layer(
                model_clone.llama_block_seq.llama_1.linear[1],
                "SwiGLU: Layer 1",
                SwigluPath.FORWARD,
            )
            hook_activ_swiglu_2 = register_layer(
                model_clone.linear[1], "SwiGLU: Layer 2", SwigluPath.FORWARD
            )
        elif self.track_direction == SwigluPath.BACKWARD:
            hook_activ_swiglu_0 = register_layer(
                model_clone.llama_block_seq.llama_0.linear[1],
                "SwiGLU: Layer 0",
                SwigluPath.BACKWARD,
            )
            hook_activ_swiglu_1 = register_layer(
                model_clone.llama_block_seq.llama_1.linear[1],
                "SwiGLU: Layer 1",
                SwigluPath.BACKWARD,
            )
            hook_activ_swiglu_2 = register_layer(
                model_clone.linear[1], "SwiGLU: Layer 2", SwigluPath.BACKWARD
            )
        else:
            raise TypeError(
                f"Expected track_direction to be of type 'SwigluPath', but got {type(self.track_direction).__name__}"
            )

        # train the model
        Trainer_ = Trainer(TRAIN_CONFIG)
        Trainer_.run(model_clone, tokens, hide_progress=True)

        # computing saturations for swiglu layers
        if self.track_direction == SwigluPath.BACKWARD:
            saturation_swiglu_0 = compute_saturation("SwiGLU: Layer 0")
            print(f"SwiGLU: Layer 0 | Saturation: {saturation_swiglu_0}")
            saturation_swiglu_1 = compute_saturation("SwiGLU: Layer 1")
            print(f"SwiGLU: Layer 1 | Saturation: {saturation_swiglu_1}")
            saturation_swiglu_2 = compute_saturation("SwiGLU: Layer 2")
            print(f"SwiGLU: Layer 2 | Saturation: {saturation_swiglu_2}")

        legends = []

        plot_layer("SwiGLU: Layer 0", "black")
        legends.append("SwiGLU: Layer 0")
        hook_activ_swiglu_0.remove()

        plot_layer("SwiGLU: Layer 1", "red")
        legends.append("SwiGLU: Layer 1")
        hook_activ_swiglu_1.remove()

        plot_layer("SwiGLU: Layer 2", "green")
        legends.append("SwiGLU: Layer 2")
        hook_activ_swiglu_2.remove()

        plt.legend(legends)
        plt.show()
