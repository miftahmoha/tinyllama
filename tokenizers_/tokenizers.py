from torch import Tensor
import torch


class CharacterTokenizer:
    def __init__(self):
        self.vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?.,:;'\"\nʼ"
        self.encode = {char: tok for tok, char in enumerate(self.vocab)}
        self.decode = {tok: char for tok, char in enumerate(self.encode)}

    def tokenize(self, string: str):
        return torch.tensor([self.encode[i] for i in string], dtype=torch.long)

    def untokenize(self, tokens: Tensor):
        return "".join([self.decode[i] for i in tokens.tolist()])

    def add_eos_tokens(self, eos_token: str):
        self.vocab += eos_token
