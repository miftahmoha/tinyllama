import torch
from torch import Tensor


class CharacterTokenizer:
    def __init__(self, eos_char="#"):
        self.eos_char = eos_char

        self.vocab = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?.,:;'\"\nʼ"
            + eos_char
        )

        # encode
        self.encode = {char: tok for tok, char in enumerate(self.vocab)}
        # decode
        self.decode = {tok: char for tok, char in enumerate(self.encode)}

    def tokenize(self, string: str):
        return torch.tensor([self.encode[i] for i in string], dtype=torch.long)

    def untokenize(self, tokens: Tensor):
        return "".join([self.decode[i] for i in tokens.tolist()])

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def eos_token(self):
        return self.encode[self.eos_char]
