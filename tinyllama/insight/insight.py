from abc import ABC, abstractmethod

from torch import Tensor

from tinyllama.models import Llama
from tinyllama.training import TrainConfig


class Insight(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, model: Llama, tokens: Tensor, TRAIN_CONFIG: TrainConfig):
        raise NotImplementedError
