from typing import Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor

from config import train_config, model_config


def get_batches(
    tokens: Tensor,
    context_window: int,
    batch_size: int,
    split: str = "train",
):
    """
    Selects random batches and returns them.

    :param tokens: Tokens (tokenized input corpus)
    :type tokens: Tensor
    :param config: Configuration file containing model hyperparameters
    :type config: Dict
    :param split: Train or test set to get batches from
    :type split: str
    """

    context_window = model_config["context_window"]

    train = tokens[: int(0.8 * len(tokens))]
    val = tokens[int(0.8 * len(tokens)) : int(0.9 * len(tokens))]
    test = tokens[int(0.9 * len(tokens)) :]

    batch_data = train
    if split == "val":
        batch_data = val
    if split == "test":
        batch_data = test

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    x = torch.stack([batch_data[i : i + context_window] for i in ix])
    y = torch.stack([batch_data[i + 1 : i + context_window + 1] for i in ix])

    return x, y


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, tokens: Tensor, context_window: int, batch_size: int
):
    """
    Return the loss for batches in the train and validation sets.

    :param model: LLM model
    :type model: nn.Module
    :param tokens: Tokens (tokenized input corpus)
    :type tokens: Tensor
    :param config: Configuration file containing model hyperparameters
    :type config: Dict
    """

    out = {}
    model.eval()

    out = {}
    for split in ["train", "val"]:
        for i in range(10):
            losses = []
            x, y = get_batches(tokens, context_window, batch_size, split)
            _, loss = model(x, y)
            losses += [loss.cpu()]
        out[split] = np.mean(losses)

    model.train()
    return out


def train(
    model: nn.Module,
    tokens: Tensor,
    context_window: int,
    batch_size: int,
    epochs: int,
    log_interval: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[bool] = None,
    return_logs: bool = False,
    return_plot: bool = False,
    show_progress: bool = True,
):
    """
    Trains the LLM model.


    :param model: LLM model
    :type model: nn.Module
    :param tokens: Tokens (tokenized input corpus)
    :type tokens: Tensor
    :param config: Configuration file containing model hyperparameters
    :type config: Dict
    :param optimizer: Optimizer for LLM model
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Scheduler for step size
    :type scheduler: torch.optim.Scheduler
    :param return_logs: Activates logs for training
    :type return_logs: Boolean
    :param return_plot: Returns loss plot
    :type return_plot: Boolean
    :param show_progress: Activates progress bar
    :type show_progress: Boolean
    """

    losses = []
    # x, y = get_batches(tokens, config, split="train")
    x, y = get_batches(tokens, context_window, batch_size, split="train")

    for epoch in tqdm(range(epochs), disable=not show_progress):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % log_interval == 0:
            out = evaluate_loss(model, tokens, context_window, batch_size)
            losses += [out]
            if return_logs:
                print(
                    f'Epoch: {epoch} | training loss: {out["train"]} | validation loss: {out["val"]}'
                )
    # print(f'val loss: {losses[-5]["val"]}')

    if return_plot:
        pd.DataFrame(losses).plot()
        plt.show()
    else:
        return losses
