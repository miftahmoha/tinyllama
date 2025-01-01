from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from tinyllama.insight import Insight
from tinyllama.models import Llama
from tinyllama.training import TrainConfig, Trainer


class GdrInsight(Insight):
    def __init__(self, *, num_iters: int, num_params_to_track: int):
        self.num_iters = num_iters
        self.num_params_to_track = num_params_to_track

    def run(self, model: Llama, tokens: torch.Tensor, TRAIN_CONFIG: TrainConfig):
        model_clone = model.clone()

        Trainer_ = Trainer(TRAIN_CONFIG)
        # necessary initial training job, data can't be found from deepcopy otherwise
        Trainer_.run(model_clone, tokens)

        gd_records = defaultdict(list)
        for _ in tqdm(range(self.num_iters), colour="green"):
            # retrieve data for each param before training
            for count, elem in enumerate(model_clone.named_parameters()):
                if elem[1].grad is not None:
                    gd_records[elem[0]].append(1 / elem[1].detach().cpu().std())
                if count > self.num_params_to_track:
                    break

            # [TODO] cache `DISABLE_TQDM`, then disable run
            Trainer_.run(model_clone, tokens)

            # compute gdratio (lr*grad)/data for each param after training
            for count, elem in enumerate(model_clone.named_parameters()):
                if elem[1].grad is not None:
                    gd_records[elem[0]][-1] = (
                        gd_records[elem[0]][-1]
                        * elem[1].grad.detach().cpu().std()
                        * TRAIN_CONFIG["lr"]
                    )
                if count > self.num_params_to_track:
                    break

        for name, gdr_list in gd_records.items():
            name = ".".join(name.split(".")[-2:])
            # legends.append(f"Param name: {name}")
            plt.plot(gdr_list)

        # recommended threshold
        plt.axhline(y=1e-3, color="r", linestyle="--")
        plt.title("Gradient/Data ratio across multiple runs")
        plt.show()
