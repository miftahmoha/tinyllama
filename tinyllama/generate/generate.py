from tqdm import tqdm

import torch
from torch import nn

from ..tokenizers import CharacterTokenizer
from ..models import Llama

# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate(
    model: Llama,
    corpus: str,
    num_tokens: int = 50,
    tokenizer: CharacterTokenizer = CharacterTokenizer(),
    kv_cache: bool = True,
):
    """
    Generates random samples from LLM model.
    """

    tokens_in = tokenizer.tokenize(corpus).clone().detach().view((1, -1)).to(device)

    tokens_out = torch.Tensor([]).to(device)

    for _ in tqdm(range(num_tokens)):
        logits = model(tokens_in, kv_cache=kv_cache)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        tokens_out = torch.cat((tokens_out, next_token), dim=0)

        if kv_cache:
            tokens_in = next_token
        else:
            tokens_in = torch.cat((tokens_in, next_token), dim=1)

    output_text = tokenizer.untokenize(tokens_out.view(-1))

    return output_text
