from collections import OrderedDict
import json

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import numpy as np

from normalization import RMSnorm
from encoding import get_rotary_matrix
from activations import SwiGLU

from training import train
from diagnosis import lr_diagnose, swiglu_diagnose, gradient_diagnose, gdratio_diagnose
from gptuner import gptune
from config import train_config, model_config


class roPEAttentionHead(nn.Module):
    def __init__(self, context_window: int, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(context_window, emb_dim)
        self.rms = RMSnorm((context_window, emb_dim))

        self.w_q = nn.Linear(emb_dim, emb_dim)
        self.w_k = nn.Linear(emb_dim, emb_dim)
        self.w_v = nn.Linear(emb_dim, emb_dim)

        self.R = get_rotary_matrix(context_window, emb_dim)

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
            attn_weights = torch.bmm(q_rot, k_rot.transpose(-2, -1)) / np.sqrt(
                self.emb_dim
            )
            attn_weights = torch.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations


class roPEMultiAttentionHead(nn.Module):
    def __init__(self, context_window: int, emb_dim: int, n_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [roPEAttentionHead(context_window, emb_dim) for i in range(n_heads)]
        )
        self.linear = nn.Linear(emb_dim * n_heads, emb_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor):
        x = [head(x) for head in self.heads]
        x = torch.cat(x, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class LlamaBlock(nn.Module):
    def __init__(self, vocab_size: int, context_window: int, emb_dim, n_heads: int):
        super().__init__()
        self.rms = RMSnorm((context_window, emb_dim))
        self.multi_attn_head = roPEMultiAttentionHead(context_window, emb_dim, n_heads)
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            SwiGLU(emb_dim),
        )
        self.last_linear = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.multi_attn_head(x)  # (B, C, emb_dim)

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.linear(x)  # (B, C, emb_dim)

        return x


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size: int = model_config["vocab_size"],
        context_window: int = model_config["context_window"],
        emb_dim: int = model_config["emb_dim"],
        n_heads: int = model_config["n_heads"],
        n_blocks: int = model_config["n_blocks"],
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.llama_block_seq = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"llama_{i}",
                        LlamaBlock(vocab_size, context_window, emb_dim, n_heads),
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
        self.tokens = None
        self.context_window = context_window

    def forward(self, x: Tensor, targets: Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)

        x = self.llama_block_seq(x)  # (B, C, emb_dim)

        x = self.linear(x)  # (B, C, emb_dim)

        logits = self.last_linear(x)  # (B, C, vocab_size)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            return logits, loss

    def setup_tokens(self, tokens: Tensor):
        self.tokens = tokens

    def train_model(
        self,
        batch_size: int = train_config["batch_size"],
        epochs: int = train_config["epochs"],
        log_interval: int = train_config["log_interval"],
    ):
        if self.tokens is not None:
            optimizer = torch.optim.Adam(self.parameters())
            train(
                self,
                self.tokens,
                self.context_window,
                batch_size,
                epochs,
                log_interval,
                optimizer,
            )
        else:
            raise ValueError(
                f"You must initialize model {type(self)} with model.setup_tokens(tokens)."
            )

    def diagnose_model(self, diagnosis_choice: str = "lr", **kwargs):
        if self.tokens is not None:
            match diagnosis_choice:
                case "lr":
                    lr_diagnose(self, self.tokens, self.context_window, **kwargs)
                case "swiglu":
                    swiglu_diagnose(self, self.tokens, self.context_window, **kwargs)
                case "gradients":
                    gradient_diagnose(self, self.tokens, self.context_window, **kwargs)
                case "gdratio":
                    gdratio_diagnose(self, self.tokens, self.context_window, **kwargs)
                case "gptune":
                    gptune(self, self.tokens, self.context_window, **kwargs)
                case _:
                    raise ValueError("This is not a valid argument.")
        else:
            raise ValueError(
                f"You must initialize model {type(self)} with model.setup_tokens(tokens)."
            )
