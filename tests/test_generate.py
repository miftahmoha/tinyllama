from copy import deepcopy

import pytest
import torch
from tinyllama import Llama
from tinyllama.tokenizers import CharacterTokenizer


# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = CharacterTokenizer()

model_with_kvcache = Llama(
    context_window=10, emb_dim=1, n_heads=1, n_blocks=1, vocab_size=tokenizer.vocab_size
)
model_without_kvcache = deepcopy(model_with_kvcache)


@pytest.mark.parametrize(
    "tokens_in_wkv, tokens_in_nkv",
    [
        (
            torch.tensor([1, 2, 3]).unsqueeze(0).to(device),
            torch.tensor([1, 2, 3]).unsqueeze(0).to(device),
        ),
        (
            torch.tensor([1, 2, 3, 4]).unsqueeze(0).to(device),
            torch.tensor([4]).unsqueeze(0).to(device),
        ),
        (
            torch.tensor([1, 2, 3, 4, 5]).unsqueeze(0).to(device),
            torch.tensor([5]).unsqueeze(1).to(device),
        ),
    ],
)
def test_kvcache(tokens_in_wkv, tokens_in_nkv):
    # set eval mode
    model_with_kvcache.eval()
    model_without_kvcache.eval()

    with torch.no_grad():
        logits_wkv = model_with_kvcache(tokens_in_wkv, kv_cache=True)
        logits_nkv = model_with_kvcache(tokens_in_nkv, kv_cache=False)

    assert torch.allclose(logits_wkv[:, -1, :], logits_nkv[:, -1, :], atol=1e-1)


def test_clear_kvcache():
    model_with_kvcache.clear_kv_cache()
    for llama_block in model_with_kvcache.llama_block_seq:
        for attention_head in llama_block.multi_attn_head.heads:  # type: ignore
            assert attention_head.cache["k_rot"].numel() == 0
            assert attention_head.cache["v"].numel() == 0
