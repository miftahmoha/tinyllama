from tinyllama.tokenizers import CharacterTokenizer


string = "This is a sentence to test CharacterTokenizer"


def regenerate_string(string):
    tokenizer = CharacterTokenizer()
    tokens = tokenizer.tokenize(string)
    return tokenizer.untokenize(tokens)


def test_tokenizer():
    regenerated_string = regenerate_string(
        "This is a sentence to test CharacterTokenizer"
    )
    assert string == regenerated_string
