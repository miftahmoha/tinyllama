from typing import List, Dict, Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from tokenizers_ import CharacterTokenizer

import torch
from torch.nn import functional as F
from torch import Tensor

# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_pdf_text(pdf_docs: List[PdfReader]):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_batches(
    tok_text: Tensor,
    config: Dict,
    split: str = "train",
):
    context_window = config["context_window"]

    train = tok_text[: int(0.8 * len(tok_text))]
    val = tok_text[int(0.8 * len(tok_text)) : int(0.9 * len(tok_text))]
    test = tok_text[int(0.9 * len(tok_text)) :]

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
def evaluate_loss(model, tok_text: Tensor, config: Dict):
    out = {}
    model.eval()

    out = {}
    for split in ["train", "val"]:
        for i in range(10):
            losses = []
            x, y = get_batches(tok_text, config, split)
            _, loss = model(x, y)
            losses += [loss.cpu()]
        out[split] = np.mean(losses)

    model.train()
    return out


def train(
    model,
    tok_text: str,
    config: Dict,
    optimizer,
    scheduler: Optional[bool] = None,
    return_logs: Optional[bool] = False,
    return_plot: Optional[bool] = False,
):
    losses = []
    x, y = get_batches(tok_text, config, split="train")

    for epoch in tqdm(range(config["epochs"])):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        if epoch % config["log_interval"] == 0:
            out = evaluate_loss(model, tok_text, config)
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
    untok_input: str, tokenizer: CharacterTokenizer, model, config: Dict
):
    tok_input = (
        torch.tensor(tokenizer.tokenize(untok_input), dtype=torch.long)
        .view((1, -1))
        .to(device)
    )

    # number of tokens to generate
    num_tokens = 50

    tok_output = torch.Tensor([]).to(device)
    for token in tqdm(range(num_tokens)):
        logits = model(tok_input)
        probs = F.softmax(logits[:, -1, :], dim=-1)

        next_tok = torch.multinomial(probs, num_samples=1)
        tok_output = torch.cat((tok_output, next_tok), dim=0)
        tok_input = torch.cat((tok_input, next_tok), dim=1)

    output_text = tokenizer.untokenize(tok_output.view(-1))
    return output_text
