from tinyllama.tokenizers import CharacterTokenizer

string = "This is a sentence to test CharacterTokenizer"

tokenizer = CharacterTokenizer()
tokens = tokenizer.tokenize(string)
regenerated_string = tokenizer.untokenize(tokens)


def test_tokenizer():
    assert string == regenerated_string
