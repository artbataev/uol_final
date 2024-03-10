# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

from typing import Any, Dict, List, Union

import jiwer
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class ExtendedWordErrorRate(Metric):
    """
    This class is an extended implementation for Word Error Rage metric,
    derived from torchmetrics.text.WordErrorRate
    https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/text/wer.py
    and jiwer docs https://jitsi.github.io/jiwer/reference/measures/
    Computes WER along with its components (deletions, insertions, substitutions)
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    errors: Tensor
    total: Tensor
    del_: Tensor
    ins: Tensor
    sub: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # add states for WER (errors and number of observed targets)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        # add states for WER components
        self.add_state("del_", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("sub", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("ins", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        """Update state, computing all components of WER"""
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]
        measures = jiwer.compute_measures(truth=target, hypothesis=preds)
        # update states for WER
        self.errors += measures["deletions"] + measures["insertions"] + measures["substitutions"]
        self.total += measures["hits"] + measures["deletions"] + measures["substitutions"]
        # update WER components
        self.del_ += measures["deletions"]
        self.ins += measures["insertions"]
        self.sub += measures["substitutions"]

    def compute(self) -> Dict[str, Tensor]:
        # return metric as dict
        return dict(
            wer=self.errors / self.total,
            deletions=self.del_ / self.total,
            insertions=self.ins / self.total,
            substitutions=self.sub / self.total,
        )
