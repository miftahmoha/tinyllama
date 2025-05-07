from os import getenv

import torch

# set device to gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set environment vars
DISABLE_PROGRESS: bool = bool(int(getenv("DISABLE_PROGRESS", 0)))
DISABLE_LOGS: bool = bool(int(getenv("DISABLE_LOGS", 1)))
QUIET_STAN: bool = bool(int(getenv("QUIET_STAN", 1)))
DISABLE_PLOT: bool = bool(int(getenv("DISABLE_PLOT", 0)))


def compute_statistics(tensor: torch.Tensor, precision: float = 1e-4):
    num_close_to_zero = torch.sum(torch.abs(tensor) < precision).item()
    total_elements = tensor.numel()
    return (
        torch.mean(tensor).item(),
        torch.std(tensor).item(),
        num_close_to_zero / total_elements,
    )
