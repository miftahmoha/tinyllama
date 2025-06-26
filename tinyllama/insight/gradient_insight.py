import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from tinyllama.globals import compute_statistics
from tinyllama.models import Llama


class GradInsight:
    def __init__(self, *, num_params_to_track: int, show_params_name: bool = False):
        self.num_params_to_track = num_params_to_track
        self.show_params_name = show_params_name

    def run(self, model: Llama):
        legends: list[str] = []
        gradients: torch.Tensor = torch.tensor([], device="cuda")

        for count, elem in tqdm(
            enumerate(model.named_parameters()), total=self.num_params_to_track
        ):
            if elem[1].grad is not None:
                # access the gradients for the parameter
                param_gradients = elem[1].grad.cpu()

                hy, hx = torch.histogram(param_gradients, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                name = ".".join(elem[0].split(".")[-2:])

                if self.show_params_name:
                    legends.append(f"{name}")

                gradients = torch.cat([gradients, elem[1].grad.flatten()])

                if count > self.num_params_to_track:
                    break

        mean, std, saturation = compute_statistics(gradients.cpu())
        print(
            f"Gradients | Mean: {mean} | Standard deviation: {std} | Saturation: {saturation}"
        )

        plt.title("Gradient density")
        plt.legend(legends)
        plt.show()
