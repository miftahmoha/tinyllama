import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm

from ..models import Llama


def get_batches(
    tokens: Tensor,
    context_window: int,
    batch_size: int,
    split: str = "train",
) -> tuple[Tensor, Tensor]:
    """
    Selects random batches and returns them.
    """

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
) -> dict[str, float]:
    """
    Return the loss for batches in the train and validation sets.
    """

    out: dict[str, float] = {}

    model.eval()

    for split in ["train", "val"]:
        for i in range(10):
            losses = []
            x, y = get_batches(tokens, context_window, batch_size, split)
            _, loss = model(x, y)
            losses += [loss.cpu()]
        out[split] = np.mean(losses)

    model.train()
    return out


class TrainConfig:
    def __init__(self, **kwargs):
        try:
            self.batch_size = kwargs.pop("batch_size")
            self.epochs = kwargs.pop("epochs")
            self.log_interval = kwargs.pop("log_interval", 10)
        except KeyError as e:
            print(f"Missing keyword argument {e}=... in TrainConfig")
            raise SystemExit

    def __getitem__(self, name: str):
        return self.__getattribute__(name)

    def __setitem__(self, name: str, value: int):
        self.__setattr__(name, value)


class Trainer:
    """Base class for training a Llama model"""

    TRAIN_CONFIG: TrainConfig

    def __init__(self, TRAIN_CONFIG: TrainConfig):
        self.TRAIN_CONFIG = TRAIN_CONFIG

    def run(
        self,
        model: Llama,
        tokens: Tensor,
        show_logs: bool = False,
        hide_progress: bool = False,
        scheduler=None,
    ) -> list:
        optimizer = torch.optim.Adam(model.parameters())

        losses = []
        x, y = get_batches(
            tokens, model.context_window, self.TRAIN_CONFIG.batch_size, split="train"
        )

        for epoch in tqdm(range(self.TRAIN_CONFIG.epochs), disable=hide_progress):
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if epoch % self.TRAIN_CONFIG.log_interval == 0:
                out = evaluate_loss(
                    model, tokens, model.context_window, self.TRAIN_CONFIG.batch_size
                )
                losses += [out]
                if show_logs:
                    print(
                        f'Epoch: {epoch} | training loss: {out["train"]} | validation loss: {out["val"]}'
                    )
        # print(f'val loss: {losses[-5]["val"]}')

        return losses
