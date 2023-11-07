from typing import List, Dict, Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from tokenizers_ import CharacterTokenizer

import torch
from torch import nn
from torch import Tensor

# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pdf_text(pdf_docs: List[PdfReader]):
    """
    Reads a list of PDFs.

    :param pdf_docs: List containing the PDFs
    :type pdf_docs: List[PdfReader]
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_batches(
    tokens: Tensor,
    config: Dict,
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

    context_window = config["context_window"]

    train = tokens[: int(0.8 * len(tokens))]
    val = tokens[int(0.8 * len(tokens)) : int(0.9 * len(tokens))]
    test = tokens[int(0.9 * len(tokens)) :]

    batch_data = train
    if split == "val":
        batch_data = val
    if split == "test":
        batch_data = test

    ix = torch.randint(
        0, batch_data.size(0) - context_window - 1, (config["batch_size"],)
    )

    x = torch.stack([batch_data[i : i + context_window] for i in ix])
    y = torch.stack([batch_data[i + 1 : i + context_window + 1] for i in ix])

    return x, y


@torch.no_grad()
def evaluate_loss(model: nn.Module, tokens: Tensor, config: Dict):
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
            x, y = get_batches(tokens, config, split)
            _, loss = model(x, y)
            losses += [loss.cpu()]
        out[split] = np.mean(losses)

    model.train()
    return out


def train(
    model,
    tokens: str,
    config: Dict,
    optimizer,
    scheduler: Optional[bool] = None,
    return_logs: Optional[bool] = False,
    return_plot: Optional[bool] = False,
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
    x, y = get_batches(tokens, config, split="train")

    for epoch in tqdm(range(config["epochs"]), disable=not show_progress):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config["log_interval"] == 0:
            out = evaluate_loss(model, tokens, config)
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


def simple_makemore(
    corpus: str, tokenizer: CharacterTokenizer, model: nn.Module, config: Dict
):
    """
    Generates random samples from LLM model.

    :param corpus: Corpus containg text
    :type corpus: str
    :param tokenizer: Tokenizer
    :type tokenizer: CharacterTokenizer
    :param model: LLM model
    :type model: nn.Module
    :param config: Configuration file
    :type config: Dict
    """

    tok_input = (
        torch.tensor(tokenizer.tokenize(corpus), dtype=torch.long)
        .view((1, -1))
        .to(device)
    )

    # number of tokens to generate
    num_tokens = 50

    tok_output = torch.Tensor([]).to(device)
    for token in tqdm(range(num_tokens)):
        logits = model(tok_input)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_tok = torch.multinomial(probs, num_samples=1)
        tok_output = torch.cat((tok_output, next_tok), dim=0)
        tok_input = torch.cat((tok_input, next_tok), dim=1)

    output_text = tokenizer.untokenize(tok_output.view(-1))
    return output_text
