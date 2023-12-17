from abc import ABC, abstractmethod

import torch


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, corpus: str):
        raise NotImplementedError

    @abstractmethod
    def untokenize(self, tokens: torch.Tensor):
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    def __init__(self):
        self.vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?.,:;'\"\nʼ"
        self.encode = {char: tok for tok, char in enumerate(self.vocab)}
        self.decode = {tok: char for tok, char in enumerate(self.encode)}

    def tokenize(self, string: str):
        return torch.tensor([self.encode[i] for i in string], dtype=torch.long)

    def untokenize(self, tokens: torch.tensor):
        return "".join([self.decode[i] for i in tokens.tolist()])
