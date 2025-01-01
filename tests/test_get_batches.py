import pytest
import torch

from tinyllama.training import get_batches

# set device to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_batches(context_window=32, batch_size=2):
    tokens = torch.randint(1, 64, (5 * context_window,)).to(device)

    x, y = get_batches(tokens, context_window, batch_size)
    return x, y


def test_offset():
    x, y = init_batches()
    assert x[0][1] == y[0][0]


@pytest.mark.parametrize(
    "context_window, batch_size",
    [(2, 1), (16, 4), (64, 16), (128, 32)],
)
def test_context_window(context_window, batch_size):
    x, y = init_batches(context_window, batch_size)
    assert x.shape[1] == context_window and y.shape[1] == context_window


@pytest.mark.parametrize(
    "context_window, batch_size",
    [(2, 1), (16, 4), (64, 16), (128, 32)],
)
def test_batch_size(context_window, batch_size):
    x, y = init_batches(context_window, batch_size)
    assert x.shape[0] == batch_size and y.shape[0] == batch_size
