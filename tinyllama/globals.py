from os import getenv
from typing import cast

import torch

# set device to gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set environment vars
# [TODO] doesn't seem to work
DISABLE_LOGS: bool = cast(bool, getenv("DISABLE_LOGS", 1))
DISABLE_TQDM: bool = cast(bool, getenv("DISABLE_TQDM", 0))
