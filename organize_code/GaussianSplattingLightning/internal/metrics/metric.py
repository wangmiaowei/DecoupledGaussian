from typing import Tuple, Dict, Any
import torch
from internal.configs.instantiate_config import InstantiatableConfig


class MetricImpl(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str, pl_module):
        pass

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        """
        :return:
            The first dict: contains the metric values.
                The `backward()` only will be invoked for the one with key `loss`.
                All other values are only for logging.
            The second dict: indicates whether the metric value should be shown on progress bar
        """

        return self.get_validate_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        pass

    def on_parameter_move(self, *args, **kwargs):
        pass


class Metric(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> MetricImpl:
        pass
