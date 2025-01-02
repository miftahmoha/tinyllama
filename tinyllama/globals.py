from os import getenv

import torch

# set device to gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set environment vars
DISABLE_LOGS: bool = bool(int(getenv("DISABLE_LOGS", 1)))
DISABLE_TQDM: bool = bool(int(getenv("DISABLE_TQDM", 0)))
