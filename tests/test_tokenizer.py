import torch
from tokenizers_ import CharacterTokenizer

string = "I love tennis!"
tokenizer = CharacterTokenizer(string)
tok_string = torch.tensor(tokenizer.tokenize(string), dtype=torch.int8)
untok_string = tokenizer.untokenize(tok_string)


def test_tokenizer():
    assert string == untok_string
