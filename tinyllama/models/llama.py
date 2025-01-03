import math
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllama.activations import SwiGLU
from tinyllama.encoding import get_rotary_matrix
from tinyllama.globals import DEVICE
from tinyllama.normalization import RMSnorm


def modified_dot_product_attention(
    self: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
) -> Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_bias = attn_bias.to(DEVICE)
    attn_weight += attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
    return attn_weight @ value


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
            "k_rot": torch.empty(1, 0, emb_dim, requires_grad=False).to(DEVICE),
            "v": torch.empty(1, 0, emb_dim, requires_grad=False).to(DEVICE),
        }

    def forward(self, x: Tensor, kv_cache: bool, return_attn_weights: bool = False):
        _, C, _ = x.shape

        # computes queries, keys & values
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # using roPE
        q_rot = torch.bmm(q.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)
        k_rot = torch.bmm(k.transpose(0, 1), self.R[:C].transpose(1, 2)).transpose(0, 1)

        if kv_cache:
            self.cache["k_rot"] = torch.cat((self.cache["k_rot"], k_rot), dim=1)
            self.cache["v"] = torch.cat((self.cache["v"], v), dim=1)

            if q_rot.size(1) > 1:
                activations = modified_dot_product_attention(
                    self,
                    q_rot,
                    self.cache["k_rot"],
                    self.cache["v"],
                    dropout_p=0.2,
                    is_causal=True,
                )
            else:
                activations = modified_dot_product_attention(
                    self,
                    q_rot,
                    self.cache["k_rot"],
                    self.cache["v"],
                    dropout_p=0.2,
                    is_causal=False,
                )
        else:
            activations = modified_dot_product_attention(
                self, q_rot, k_rot, v, dropout_p=0.2, is_causal=True
            )

        # shouldn't work with kv_cache, but not needed
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
        vocab_size: int,
    ):
        super().__init__()
        self.context_window = context_window
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.gq_ratio = gq_ratio
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
        self.to(DEVICE)

    def forward(self, x: Tensor, targets: Tensor = None, kv_cache: bool = False):  # type: ignore
        x = self.embedding(x)  # (B, C, emb_dim)
        x = self.llama_block_seq((x, kv_cache))  # (B, C, emb_dim)
        x = self.linear(x[0])  # (B, C, emb_dim)
        logits = self.last_linear(x)  # (B, C, vocab_size)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

    def clear_kv_cache(self):
        for llama_block in self.llama_block_seq:
            for attention_head in llama_block.multi_attn_head.heads:  # type: ignore
                attention_head.cache["k_rot"] = torch.empty(
                    1, 0, self.emb_dim, requires_grad=False
                ).to(DEVICE)
                attention_head.cache["v"] = torch.empty(
                    1, 0, self.emb_dim, requires_grad=False
                ).to(DEVICE)

    def new(self, **kwargs):
        # get the current hyperparameters from the existing model
        current_params = {
            "context_window": self.context_window,
            "emb_dim": self.emb_dim,
            "n_heads": self.n_heads,
            "n_blocks": self.n_blocks,
            "gq_ratio": self.gq_ratio,
            "vocab_size": self.vocab_size,
        }

        # update the current hyperparameters with the new ones provided by the user
        current_params.update(kwargs)

        # create a new model with the updated hyperparameters
        return Llama(**current_params)

    def clone(self):
        return deepcopy(self)
