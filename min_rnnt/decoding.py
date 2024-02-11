import logging
from typing import List, Optional

import torch
import torch.nn as nn

from min_rnnt.modules import MinJoint, MinPredictionNetwork


class RNNTDecodingWrapper(nn.Module):
    """
    Decoding Wrapper with Greedy Decoding implementation.
    Same algorithm as used in NeMo, but simplified for easiness of customization
    Original algorithm:
    https://github.com/NVIDIA/NeMo/blob/v1.21.0/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L478
    """

    def __init__(
        self,
        prediction_network: MinPredictionNetwork,
        joint: MinJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
    ):
        super().__init__()
        self.blank_index = blank_index
        self.prediction_network = prediction_network
        self.joint = joint
        self.max_symbols_per_step = max_symbols_per_step

    def forward(
        self,
        encoder_output: torch.Tensor,  # BxDxT
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ):
        prediction_network_output, _ = self.prediction_network(
            input_prefix=targets, input_lengths=target_lengths, add_sos=True
        )
        joint_output = self.joint(
            encoder_output=encoder_output.transpose(1, 2), prediction_output=prediction_network_output
        )
        return joint_output

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,  # [B, D, T]
        encoder_lengths: torch.Tensor,
    ) -> List[List[int]]:
        encoder_output = encoder_output.permute(2, 0, 1)  # T, B, D
        max_time, batch_size, _ = encoder_output.shape
        hyps = [[] for _ in range(batch_size)]

        state = None
        device = encoder_output.device
        # we use <blank> symbols as <SOS>
        last_label = torch.full([batch_size], fill_value=self.blank_index, dtype=torch.long, device=device)

        # loop over time frames
        for time_i in range(max_time):
            encoder_vec = encoder_output[time_i]  # get current encoder vector

            symbols_added = 0
            blank_or_end_mask = time_i >= encoder_lengths  # use labels only if we are not out-of-sequence
            while not blank_or_end_mask.all() and (
                self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step
            ):
                prev_state = state  # cache previous state
                # compute prediction network with updated labels
                prediction_output, state = self.prediction_network(
                    input_prefix=last_label.unsqueeze(1),
                    input_lengths=torch.ones_like(last_label),
                    state=state,
                    add_sos=False,
                )

                # compute Joint network
                joint_output = (
                    self.joint(encoder_output=encoder_vec.unsqueeze(1), prediction_output=prediction_output)
                    .squeeze(1)
                    .squeeze(1)
                )
                # greedy: get labels with maximum probability (maximum logit id is enough)
                labels = joint_output.argmax(dim=1)

                # check if labels are blank
                blank_or_end_mask.logical_or_(labels == self.blank_index)

                if not blank_or_end_mask.all():
                    # some labels found, not all are blank
                    non_blank_indices = (~blank_or_end_mask).nonzero(as_tuple=False).squeeze(1)
                    # store found non-blank symbols
                    for i in non_blank_indices.tolist():
                        hyps[i].append(labels[i].item())
                    symbols_added += 1

                if blank_or_end_mask.any():
                    # restore prediction network state for blank labels:
                    # network state should be updated only if non-blank found
                    blank_indices = blank_or_end_mask.nonzero(as_tuple=False).squeeze(1)
                    if prev_state is not None:
                        if isinstance(state, tuple):  # Lstm
                            # restore state for found blank labels
                            for i, sub_state in enumerate(state):
                                sub_state[:, blank_indices] = prev_state[i][:, blank_indices]
                        else:
                            raise NotImplementedError
                    else:
                        # prev_state is None, start of the decoding loop
                        if isinstance(state, tuple):
                            # restore 0 initial state for found blank labels
                            for i, sub_state in enumerate(state):
                                sub_state[:, blank_indices] = 0.0
                        else:
                            raise NotImplementedError

                    labels[blank_indices] = last_label[blank_indices]
                last_label = labels.clone()
        return hyps
