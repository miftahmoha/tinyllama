import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

from normalization import RMSnorm
from pos_encoding import get_rotary_matrix


class roPeAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["context_window"], config["emb_dim"])
        # self.linear = nn.Linear(config["emb_dim"], config["vocab_size"])
        self.rms = RMSnorm((config["context_window"], config["emb_dim"]))

        self.w_q = nn.Linear(config["emb_dim"], config["emb_dim"])
        self.w_k = nn.Linear(config["emb_dim"], config["emb_dim"])
        self.w_v = nn.Linear(config["emb_dim"], config["emb_dim"])

        self.R = get_rotary_matrix(config["context_window"], config["emb_dim"])

    def forward(self, x: Tensor, return_attn_weights: bool = False):
        x = self.embedding(x)
        B, C, emb_dim = x.shape

        # computes queries, keys & values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # using roPe
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
