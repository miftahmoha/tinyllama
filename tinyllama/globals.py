from os import getenv

import torch

# set device to gpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set environment vars
IGNORE_TRAINING: bool = bool(int(getenv("IGNORE_TRAINING", 0)))
DISABLE_LOGS: bool = bool(int(getenv("DISABLE_LOGS", 1)))
DISABLE_PLOT: bool = bool(int(getenv("DISABLE_PLOT", 0)))
