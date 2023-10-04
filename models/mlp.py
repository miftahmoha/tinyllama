import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

import torch
from torch import nn
from torch.nn import functional as F

from normalization import RMSnorm
from pos_encoding import get_rotary_matrix

# Set the seed for PyTorch
torch.manual_seed(42)


class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.linear = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.ReLU(),
            nn.Linear(config["emb_dim"], config["vocab_size"]),
        )

        print(f"Parameters: \n {[m.numel() for m in self.parameters()]}")

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)
        x = self.linear(x)  # (B, C, vocab_size)
        logits = F.softmax(x, dim=-1)  # (B, C, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss
        return logits


class SimpleModel_RMS(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.linear = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.ReLU(),
            nn.Linear(config["emb_dim"], config["vocab_size"]),
        )
        self.rms = RMSnorm((config["context_window"], config["emb_dim"]))
        print(f"Parameters: \n {[m.numel() for m in self.parameters()]}")

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)
        x = self.rms(x)  # (B, C, emb_dim)
        x = self.linear(x)  # (B, C, vocab_size)
        logits = F.softmax(x, dim=-1)  # (B, C, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss
        return logits


class SimpleModel_roPE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.linear = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.ReLU(),
            nn.Linear(config["emb_dim"], config["vocab_size"]),
        )
        self.rms = RMSnorm((config["context_window"], config["emb_dim"]))
        self.R = get_rotary_matrix(config["context_window"], config["emb_dim"])
        print(f"Parameters: \n {[m.numel() for m in self.parameters()]}")

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)

        # x = torch.stack(
        # [torch.bmm(self.R, x[i].unsqueeze(-1)) for i in range(x.size(0))], dim=0
        # ).squeeze(-1)

        x = torch.bmm(x.transpose(0, 1), self.R[: x.size(1)].transpose(1, 2)).transpose(
            0, 1
        )
        x = self.rms(x)  # (B, C, emb_dim)
        x = self.linear(x)  # (B, C, vocab_size)
        logits = F.softmax(x, dim=-1)  # (B, C, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss
        return logits
