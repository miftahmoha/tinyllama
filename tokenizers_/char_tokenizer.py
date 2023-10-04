import torch


class CharacterTokenizer:
    def __init__(self, corpus):
        self.vocab = sorted(list(set(corpus)))
        self.tok = {char: tok for tok, char in enumerate(self.vocab)}
        self.untok = {tok: char for tok, char in enumerate(self.tok)}

    def tokenize(self, string: str):
        return [self.tok[i] for i in string]

    def untokenize(self, tokens: torch.tensor):
        return "".join([self.untok[i] for i in tokens.tolist()])
