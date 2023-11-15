from typing import Callable
from tqdm import tqdm

import torch
from torch import nn
from PyPDF2 import PdfReader

from tokenizers_ import Tokenizer

# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# should be integrated with a custom reader that outputs corpus as a string
def pre_process_corpus(parse_func: Callable[..., str]):
    def wrapper(*args, **kwargs):
        corpus = parse_func(*args, **kwargs)
        characters_to_keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !?.,:;'\"\nʼ"

        # initialize an empty result strings
        undesirable_characters = ""

        # iterate through each character in the input string
        for char in sorted(list(set(corpus))):
            # check if the character is in the list of characters to keep
            if char not in characters_to_keep:
                # if yes, add it to the result string
                undesirable_characters += char

        # removing undesirable characters
        for char in undesirable_characters:
            corpus = corpus.replace(char, "")
        return corpus

    return wrapper


@pre_process_corpus
def get_pdf_text(pdf_docs: list[PdfReader]):
    """
    Reads a list of PDFs.

    :param pdf_docs: List containing the PDFs
    :type pdf_docs: list[PdfReader]
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def simple_makemore(corpus: str, tokenizer: Tokenizer, model: nn.Module, config: dict):
    """
    Generates random samples from LLM model.

    :param corpus: Corpus containg text
    :type corpus: str
    :param tokenizer: Tokenizer
    :type tokenizer: CharacterTokenizer
    :param model: LLM model
    :type model: nn.Module
    :param config: Configuration file
    :type config: Dict
    """

    tok_input = (
        torch.tensor(tokenizer.tokenize(corpus), dtype=torch.long)
        .view((1, -1))
        .to(device)
    )

    # number of tokens to generate
    num_tokens = 50

    tok_output = torch.Tensor([]).to(device)
    for token in tqdm(range(num_tokens)):
        logits = model(tok_input)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_tok = torch.multinomial(probs, num_samples=1)
        tok_output = torch.cat((tok_output, next_tok), dim=0)
        tok_input = torch.cat((tok_input, next_tok), dim=1)

    output_text = tokenizer.untokenize(tok_output.view(-1))
    return output_text
