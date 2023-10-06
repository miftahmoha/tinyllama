from typing import Dict

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

from normalization import RMSnorm
from encoding import get_rotary_matrix


class roPEAttentionHead(nn.Module):
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["context_window"], config["emb_dim"])
        self.rms = RMSnorm((config["context_window"], config["emb_dim"]))

        self.w_q = nn.Linear(config["emb_dim"], config["emb_dim"])
        self.w_k = nn.Linear(config["emb_dim"], config["emb_dim"])
        self.w_v = nn.Linear(config["emb_dim"], config["emb_dim"])

        self.R = get_rotary_matrix(config["context_window"], config["emb_dim"])

    def forward(self, x: Tensor, return_attn_weights: bool = False):
        # x = self.embedding(x)
        B, C, emb_dim = x.shape

        # computes queries, keys & values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # using roPE
        q_rot = torch.bmm(q.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)
        k_rot = torch.bmm(k.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)

        activations = F.scaled_dot_product_attention(
            q_rot, k_rot, v, dropout_p=0.2, is_causal=True
        )

        if return_attn_weights:
            attn_weights = torch.bmm(q_rot, k_rot.transpose(-2, -1)) / np.sqrt(emb_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations


class roPEMultiAttentionHead(nn.Module):
    def __init__(self, config: Dict[str, int], attn_head: roPEAttentionHead):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.attn_head = roPEAttentionHead(config)

        self.heads = nn.ModuleList(
            [roPEAttentionHead(config) for i in range(self.n_heads)]
        )

        self.linear = nn.Linear(
            config["emb_dim"] * config["n_heads"], config["emb_dim"]
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor):
        x = [head(x) for head in self.heads]
        x = torch.cat(x, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        config: Dict[str, int],
        attn_head: roPEAttentionHead,
        mult_attn_head: roPEMultiAttentionHead,
    ):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["emb_dim"])

        self.rms = RMSnorm((config["context_window"], config["emb_dim"]))
        self.multi_attn_head = roPEMultiAttentionHead(
            self.config, attn_head(self.config)
        )

        self.linear = nn.Sequential(
            nn.Linear(config["emb_dim"], config["emb_dim"]),
            nn.ReLU(),
        )

        self.last_linear = nn.Linear(config["emb_dim"], config["vocab_size"])

        print(f"Parameters: \n {sum([m.numel() for m in self.parameters()])}")

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.multi_attn_head(x)  # (B, C, emb_dim)

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.linear(x)  # (B, C, emb_dim)

        x = self.last_linear(x)  # (B, C, vocab_size)

        logits = F.softmax(x, dim=-1)  # (B, C, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(
                x.view(-1, self.config["vocab_size"]), targets.view(-1)
            )
            return logits, loss

        return logits
