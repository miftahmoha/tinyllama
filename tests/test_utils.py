from utils import get_batches
import os
import torch


def test_get_batches():
    MASTER_CONFIG = {"vocab_size": None, "context_window": 5}
    len_text = 20
    tok_text = torch.randint(0, 15, (40,))

    x, y = get_batches(tok_text, 8, MASTER_CONFIG)
    assert x[0][1] == y[0][0]
