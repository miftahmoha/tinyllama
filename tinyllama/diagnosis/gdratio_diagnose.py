from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy

import torch
import matplotlib.pyplot as plt

from ..models import Llama
from ..training import Trainer, TrainConfig
from ..diagnosis import Diagnose


class GdrDiagnose(Diagnose):
    def __init__(self, *, num_iters: int, num_params_to_track: int):
        self.num_iters = num_iters
        self.num_params_to_track = num_params_to_track

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):

        legends = []

        model_clone = deepcopy(model)

        TRAIN_CONFIG["epochs"] = 1
        Trainer_ = Trainer(TRAIN_CONFIG)

        gd_records = defaultdict(list)
        for _ in tqdm(range(self.num_iters), colour="green"):
            Trainer_.run(model_clone, tokens, hide_progress=True)

            for count, elem in enumerate(model_clone.named_parameters()):
                if elem[1].grad is not None:
                    gd_records[elem[0]].append(
                        elem[1].grad.cpu().std() / elem[1].detach().cpu().std()
                    )

                if count > self.num_params_to_track:
                    break

        for name, gdr_list in gd_records.items():
            name = ".".join(name.split(".")[-2:])
            legends.append(f"Param name: {name}")
            plt.plot(gdr_list)

        plt.title("Gradient / Data ratio")
        plt.legend(legends)
        plt.show()
