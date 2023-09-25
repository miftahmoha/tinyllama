import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

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

    def forward(self, x, targets=None):
        x = self.embedding(x)  # (B, C, emb_dim)
        x = self.linear(x)  # (B, C, emb_dim)
        logits = F.softmax(x, dim=-1)  # (B, C, emb_dim)

        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss
        return logits
