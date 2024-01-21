from copy import deepcopy

import torch
import matplotlib.pyplot as plt

from ..diagnosis import Diagnose
from ..training import TrainConfig, Trainer
from ..models import Llama


# a dict to store the activations
activation = {}


def getActivation(name):
    # hook signature
    def hook(model, input, output):
        activation[name] = (
            output[0].detach() if isinstance(output, tuple) else output.detach()
        )

    return hook


class SwigluDiagnose(Diagnose):
    def __init__(self, *, num_embeddings_for_histogram: int, track_direction: str):
        self.num_embeddings_for_histogram = num_embeddings_for_histogram
        self.track_direction = track_direction

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):

        model_clone = deepcopy(model)

        # hooking activations layers
        if self.track_direction == "forward":
            hook_activ_swiglu_0 = model_clone.llama_block_seq.llama_0.linear[
                1
            ].register_forward_hook(getActivation("SwiGLU activations 0"))
            hook_activ_swiglu_1 = model_clone.llama_block_seq.llama_1.linear[
                1
            ].register_forward_hook(getActivation("SwiGLU activations 1"))
            hook_activ_swiglu_2 = model_clone.linear[1].register_forward_hook(
                getActivation("SwiGLU activations 2")
            )

        else:
            hook_activ_swiglu_0 = model_clone.llama_block_seq.llama_0.linear[
                1
            ].register_full_backward_hook(getActivation("SwiGLU activations 0"))
            hook_activ_swiglu_1 = model_clone.llama_block_seq.llama_1.linear[
                1
            ].register_full_backward_hook(getActivation("SwiGLU activations 1"))
            hook_activ_swiglu_2 = model_clone.linear[1].register_full_backward_hook(
                getActivation("SwiGLU activations 2")
            )

        # train the model
        Trainer_ = Trainer(TRAIN_CONFIG)
        Trainer_.run(model_clone, tokens)

        # sample random batches
        random_batches = torch.randint(
            low=0,
            high=Trainer_.TRAIN_CONFIG["batch_size"] - 1,
            size=(self.num_embeddings_for_histogram,),
        ).tolist()

        hy, hx = torch.histogram(
            activation["SwiGLU activations 0"][
                random_batches, self.num_embeddings_for_histogram, :
            ].cpu(),
            density=True,
        )
        plt.plot(hx[:-1].detach(), hy.detach())
        hook_activ_swiglu_0.remove()
        plt.title("Histogram for the 1st SwiGLU layer (random batches):")
        plt.show()

        hy, hx = torch.histogram(
            activation["SwiGLU activations 1"][
                random_batches, self.num_embeddings_for_histogram, :
            ].cpu(),
            density=True,
        )
        plt.plot(hx[:-1].detach(), hy.detach())
        hook_activ_swiglu_1.remove()
        plt.title("Histogram for the 2nd SwiGLU layer (random batches):")
        plt.show()

        hy, hx = torch.histogram(
            activation["SwiGLU activations 2"][
                random_batches, self.num_embeddings_for_histogram, :
            ].cpu(),
            density=True,
        )
        plt.plot(hx[:-1].detach(), hy.detach())
        hook_activ_swiglu_2.remove()
        plt.title("Histogram for the 3rd SwiGLU layer (random batches):")
        plt.show()
