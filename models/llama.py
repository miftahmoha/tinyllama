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
from config import model_config


class roPEAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            model_config["context_window"], model_config["emb_dim"]
        )
        self.rms = RMSnorm((model_config["context_window"], model_config["emb_dim"]))

        self.w_q = nn.Linear(model_config["emb_dim"], model_config["emb_dim"])
        self.w_k = nn.Linear(model_config["emb_dim"], model_config["emb_dim"])
        self.w_v = nn.Linear(model_config["emb_dim"], model_config["emb_dim"])

        self.R = get_rotary_matrix(
            model_config["context_window"], model_config["emb_dim"]
        )

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
    def __init__(self):
        super().__init__()
        self.n_heads = model_config["n_heads"]
        self.heads = nn.ModuleList([roPEAttentionHead() for i in range(self.n_heads)])
        self.linear = nn.Linear(
            model_config["emb_dim"] * model_config["n_heads"], model_config["emb_dim"]
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor):
        x = [head(x) for head in self.heads]
        x = torch.cat(x, dim=-1)

        x = self.linear(x)
        x = self.dropout(x)

        return x


class LlamaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = RMSnorm((model_config["context_window"], model_config["emb_dim"]))
        self.multi_attn_head = roPEMultiAttentionHead()
        self.linear = nn.Sequential(
            nn.Linear(model_config["emb_dim"], model_config["emb_dim"]),
            SwiGLU(model_config["emb_dim"]),
        )
        self.last_linear = nn.Linear(
            model_config["emb_dim"], model_config["vocab_size"]
        )

    def forward(self, x, targets=None):
        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.multi_attn_head(x)  # (B, C, emb_dim)

        x = self.rms(x)  # (B, C, emb_dim)
        x = x + self.linear(x)  # (B, C, emb_dim)

        return x


class Llama(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            model_config["vocab_size"], model_config["emb_dim"]
        )
        self.llama_block_seq = nn.Sequential(
            OrderedDict(
                [(f"llama_{i}", LlamaBlock()) for i in range(model_config["n_blocks"])]
            )
        )
        self.linear = nn.Sequential(
            nn.Linear(model_config["emb_dim"], model_config["emb_dim"]),
            SwiGLU(model_config["emb_dim"]),
        )
        self.last_linear = nn.Linear(
            model_config["emb_dim"], model_config["vocab_size"]
        )
        print(f"Parameters: \n {sum([m.numel() for m in self.parameters()])}")
        self.tokens = None

    def forward(self, x: Tensor, targets: Tensor = None):
        x = self.embedding(x)  # (B, C, emb_dim)

        x = self.llama_block_seq(x)  # (B, C, emb_dim)

        x = self.linear(x)  # (B, C, emb_dim)

        logits = self.last_linear(x)  # (B, C, vocab_size)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(
                logits.view(-1, model_config["vocab_size"]), targets.view(-1)
            )
            return logits, loss

    def setup_tokens(self, tokens: Tensor):
        self.tokens = tokens

    def train_model(self, train_config: dict):
        if self.tokens is not None:
            optimizer = torch.optim.Adam(self.parameters())
            train(self, self.tokens, train_config, optimizer)
        else:
            raise ValueError(
                f"You must initialize model {type(self)} with model.setup_tokens(tokens)."
            )

    def diagnose_model(self, diagnosis_choice: str = "lr"):
        if self.tokens is not None:
            match diagnosis_choice:
                case "lr":
                    lr_diagnose(self, self.tokens)
                case "forward_swiglu":
                    swiglu_diagnose(self, self.tokens, mode="forward")
                case "backward_swiglu":
                    swiglu_diagnose(self, self.tokens, mode="backward")
                case "gradients":
                    gradient_diagnose(self, self.tokens)
                case "gdratio":
                    gdratio_diagnose(self, self.tokens)
                case "gptune":
                    gptune(self, self.tokens)
                case _:
                    raise ValueError("This is not a valid argument.")
        else:
            raise ValueError(
                f"You must initialize model {type(self)} with model.setup_tokens(tokens)."
            )
