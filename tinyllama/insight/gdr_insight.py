from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from tinyllama.insight import Insight
from tinyllama.models import Llama
from tinyllama.training import TrainConfig, Trainer


class GdrInsight(Insight):
    def __init__(
        self,
        *,
        num_iters: int,
        num_params_to_track: int,
        show_params_name: bool = False,
    ):
        self.num_iters = num_iters
        self.num_params_to_track = num_params_to_track
        self.show_params_name = show_params_name

    def run(
        self,
        model: Llama,
        tokens: torch.Tensor,
        TUNE_CONFIG: TrainConfig = TrainConfig(batch_size=32, epochs=64),
        tune_on_clone: bool = False,
    ):
        legends: list[str] = []
        model_ = model.clone() if tune_on_clone else model

        Trainer_ = Trainer(TUNE_CONFIG)
        # necessary initial training job, data can't be found from deepcopy otherwise
        Trainer_.run(model_, tokens)

        gd_records = defaultdict(list)
        for _ in tqdm(range(self.num_iters), colour="green"):
            # retrieve data for each param before training
            for count, elem in enumerate(model_.named_parameters()):
                if elem[1].grad is not None:
                    gd_records[elem[0]].append(1 / elem[1].detach().cpu().std())
                if count > self.num_params_to_track:
                    break

            Trainer_.run(model_, tokens)

            # compute (lr*grad)/data for each param after training
            for count, elem in enumerate(model_.named_parameters()):
                if elem[1].grad is not None:
                    gd_records[elem[0]][-1] = (
                        gd_records[elem[0]][-1]
                        * elem[1].grad.detach().cpu().std()
                        * TUNE_CONFIG["lr"]
                    )
                if count > self.num_params_to_track:
                    break

        for name, gdr_list in gd_records.items():
            name = ".".join(name.split(".")[-2:])
            if self.show_params_name:
                legends.append(f"Param name: {name}")
            plt.plot(gdr_list)

        # recommended threshold
        plt.axhline(y=1e-3, color="r", linestyle="--")
        plt.title("Gradient over data ratio across multiple runs")
        plt.legend(legends)
        plt.show()
