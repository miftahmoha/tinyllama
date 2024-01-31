from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..activations import SwiGLU
from ..encoding import get_rotary_matrix
from ..normalization import RMSnorm
from ..tokenizers import CharacterTokenizer


# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class roPEAttentionHead(nn.Module):
    def __init__(
        self, context_window: int, emb_dim: int, w_q: nn.Linear, w_k: nn.Linear
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(context_window, emb_dim)
        self.rms = RMSnorm((context_window, emb_dim))

        self.w_q = w_q
        self.w_k = w_k
        self.w_v = nn.Linear(emb_dim, emb_dim)

        self.R = get_rotary_matrix(context_window, emb_dim)
        self.cache = {
            "keys": torch.empty(1, 0, emb_dim).cuda(),
            "values": torch.empty(1, 0, emb_dim).cuda(),
        }

    def forward(self, x: Tensor, kv_cache: bool, return_attn_weights: bool = False):
        # x = self.embedding(x)
        _, C, _ = x.shape

        # computes queries, keys & values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # using roPE
        q_rot = torch.bmm(q.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)
        k_rot = torch.bmm(k.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)

        if kv_cache:
            self.cache["keys"] = torch.cat((self.cache["keys"], k_rot), dim=1)
            self.cache["values"] = torch.cat((self.cache["values"], v), dim=1)

        activations = F.scaled_dot_product_attention(
            q_rot, k_rot, v, dropout_p=0.2, is_causal=True
        )

        if return_attn_weights:
            attn_weights = torch.bmm(q_rot, k_rot.transpose(-2, -1)) / np.sqrt(
                self.emb_dim
            )
            attn_weights = torch.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations


class roPEMultiAttentionHead(nn.Module):
    def __init__(self, context_window: int, emb_dim: int, n_heads: int, gq_ratio: int):
        super().__init__()

        self.heads = []
        for i in range(n_heads):
            if i % gq_ratio == 0:
                w_q = nn.Linear(emb_dim, emb_dim)
                w_k = nn.Linear(emb_dim, emb_dim)

            self.heads += [roPEAttentionHead(context_window, emb_dim, w_q, w_k)]
        self.heads = nn.ModuleList(self.heads)

        self.linear = nn.Linear(emb_dim * n_heads, emb_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor, kv_cache: bool):
        x = [head(x, kv_cache) for head in self.heads]
        x = torch.cat(x, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class LlamaBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_window: int,
        emb_dim: int,
        n_heads: int,
        gq_ratio: int,
    ):
        super().__init__()
        self.rms = RMSnorm((context_window, emb_dim))
        self.multi_attn_head = roPEMultiAttentionHead(
            context_window, emb_dim, n_heads, gq_ratio
        )
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            SwiGLU(emb_dim),
        )
        self.last_linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, params: tuple[Tensor, bool]):
        x, kv_cache = params

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.multi_attn_head(x, kv_cache)  # (B, C, emb_dim)

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.linear(x)  # (B, C, emb_dim)

        return x, kv_cache


class Llama(nn.Module):
    def __init__(
        self,
        *,
        context_window: int,
        emb_dim: int,
        n_heads: int,
        n_blocks: int,
        gq_ratio: int = 1,
        vocab_size: int = CharacterTokenizer().vocab_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.llama_block_seq = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"llama_{i}",
                        LlamaBlock(
                            vocab_size, context_window, emb_dim, n_heads, gq_ratio
                        ),
                    )
                    for i in range(n_blocks)
                ]
            )
        )
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            SwiGLU(emb_dim),
        )
        self.last_linear = nn.Linear(emb_dim, vocab_size)
        print(f"Parameters: \n {sum([m.numel() for m in self.parameters()])}")
        self.context_window = context_window
        self.to(device)

    def forward(self, x: Tensor, targets: Tensor = None, kv_cache: bool = False):
        x = self.embedding(x)  # (B, C, emb_dim)

        x = self.llama_block_seq((x, kv_cache))  # (B, C, emb_dim)

        x = self.linear(x[0])  # (B, C, emb_dim)

        logits = self.last_linear(x)  # (B, C, vocab_size)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss
