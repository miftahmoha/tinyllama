import torch
from torch import nn
from tqdm import tqdm

from tinyllama.globals import DEVICE
from tinyllama.models import Llama
from tinyllama.tokenizers import CharacterTokenizer


def generate(
    model: Llama,
    tokenizer: CharacterTokenizer,
    prompt: str,
    num_tokens: int = 50,
    kv_cache: bool = True,
):
    """
    Generates random samples from LLM model.
    """

    tokens_in = tokenizer.tokenize(prompt).view((1, -1)).to(DEVICE)
    tokens_out = torch.tensor([], requires_grad=False).to(DEVICE)

    for _ in tqdm(range(num_tokens)):
        # set eval mode
        model.eval()

        with torch.no_grad():
            logits = model(tokens_in, kv_cache=kv_cache)
            probs = nn.functional.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens_out = torch.cat((tokens_out, next_token), dim=0)
            tokens_in = (
                next_token if kv_cache else torch.cat((tokens_in, next_token), dim=1)
            )

    # reset train mode
    model.train()
    model.clear_kv_cache()
    output_text = tokenizer.untokenize(tokens_out.view(-1))

    return output_text
