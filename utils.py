from typing import Callable
from tqdm import tqdm

import torch
from torch import nn
from PyPDF2 import PdfReader

from tokenizers_ import Tokenizer, CharacterTokenizer
from models import Llama

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


def simple_makemore_(
    model: Llama,
    corpus: str = "",
    num_tokens: int = 50,
    tokenizer: Tokenizer = CharacterTokenizer(),
):
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

    tokens_in = (
        torch.tensor(tokenizer.tokenize(corpus), dtype=torch.long)
        .view((1, -1))
        .to(device)
    )

    tokens_out = torch.Tensor([]).to(device)

    for _ in tqdm(range(num_tokens)):
        logits = model(tokens_in)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens_out = torch.cat((tokens_out, next_token), dim=0)
        tokens_in = torch.cat((tokens_in, next_token), dim=1)

    output_text = tokenizer.untokenize(tokens_out.view(-1))

    return output_text


def generate(
    model: Llama,
    corpus: str = "",
    num_tokens: int = 50,
    tokenizer: Tokenizer = CharacterTokenizer(),
    inference_mode: bool = True,
):
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

    tokens_in = (
        torch.tensor(tokenizer.tokenize(corpus), dtype=torch.long)
        .view((1, -1))
        .to(device)
    )

    tokens_out = torch.Tensor([]).to(device)

    for _ in tqdm(range(num_tokens)):
        logits = model(tokens_in, inference_mode=inference_mode)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens_out = torch.cat((tokens_out, next_token), dim=0)
        if inference_mode:
            tokens_in = next_token
        else:
            tokens_in = torch.cat((tokens_in, next_token), dim=1)

    output_text = tokenizer.untokenize(tokens_out.view(-1))

    return output_text
