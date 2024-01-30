import torch
from torch import nn
from tqdm import tqdm

from ..models import Llama
from ..tokenizers import CharacterTokenizer


# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate(
    model: Llama,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_tokens: int = 50,
    kv_cache: bool = True,
):
    """
    Generates random samples from LLM model.
    """

    tokens_in = tokenizer.tokenize(prompt).clone().detach().view((1, -1)).to(device)

    tokens_out = torch.Tensor([]).to(device)

    for _ in tqdm(range(max_tokens)):
        logits = model(tokens_in, kv_cache=kv_cache)
        probs = nn.functional.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        if next_token == tokenizer.eos_token:
            print("eos_token is reached!")
            break

        tokens_out = torch.cat((tokens_out, next_token), dim=0)

        tokens_in = (
            next_token if kv_cache else torch.cat((tokens_in, next_token), dim=1)
        )

    output_text = tokenizer.untokenize(tokens_out.view(-1))

    return output_text
