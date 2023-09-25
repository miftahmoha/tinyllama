import numpy as np
import torch
from tokenizer import tokenizer

string = "I love tennis!"
tokenizer = tokenizer(string)
tok_string = torch.tensor(tokenizer.tokenize(string), dtype=torch.int8)
untok_string = tokenizer.untokenize(tok_string)


def test_tokenizer():
    assert string == untok_string
