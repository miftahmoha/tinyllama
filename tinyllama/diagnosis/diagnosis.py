from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from ..models import Llama
from ..training import TrainConfig


class Diagnose(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, model: Llama, tokens: Tensor, TRAIN_CONFIG: TrainConfig):
        raise NotImplementedError
