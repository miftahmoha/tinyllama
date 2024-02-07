import torch
from torch import Tensor


# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CharacterTokenizer:
    def __init__(self):
        self.vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !#$%&'[]()*+,-./:;<=>?@^_`{|}~\t\n\r"

        # encode
        self.encode = {char: tok for tok, char in enumerate(self.vocab)}
        # decode
        self.decode = {tok: char for tok, char in enumerate(self.encode)}

    def tokenize(self, string: str):
        return torch.tensor(
            [self.encode[i] for i in string], dtype=torch.long, requires_grad=False
        ).to(device)

    def untokenize(self, tokens: Tensor):
        return "".join([self.decode[i] for i in tokens.tolist()])

    @property
    def vocab_size(self):
        return len(self.vocab)
