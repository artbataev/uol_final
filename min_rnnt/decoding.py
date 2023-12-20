import logging
from typing import List, Optional

import torch
import torch.nn as nn

from min_rnnt.modules import MinJoint, MinPredictionNetwork


class RNNTDecodingWrapper(nn.Module):
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
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ):
        prediction_network_output = self.prediction_network(
            input_prefix=targets, input_lengths=target_lengths, add_sos=True
        )
        joint_output = self.joint(encoder_output=encoder_output, prediction_output=prediction_network_output)
        return joint_output

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,  # [B, D, T]
        encoder_lengths: torch.Tensor,
    ) -> List[List[int]]:
        encoder_output = encoder_output.permute(2, 0, 1)  # T, B, D
        max_time, batch_size, _ = encoder_output.shape
        # logging.warning(f"B {batch_size}, T {max_time}")
        hyps = [[] for _ in range(batch_size)]

        state = None
        device = encoder_output.device
        last_label = torch.full([batch_size], fill_value=self.blank_index, dtype=torch.long, device=device)

        for time_i in range(max_time):
            encoder_vec = encoder_output[time_i]

            all_blank_found = False
            symbols_added = 0
            blank_mask = time_i >= encoder_lengths
            while not all_blank_found and (
                self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step
            ):
                # blank_mask_prev = blank_mask.clone()
                prev_state = state
                prediction_output, state = self.prediction_network(
                    input_prefix=last_label.unsqueeze(1),
                    input_lengths=torch.ones_like(last_label),
                    state=state,
                    add_sos=False,
                )
                # logging.warning(f"prediction_output {prediction_output.shape}")
                # logging.warning(f"{state[0].shape}")

                joint_output = (
                    self.joint(encoder_output=encoder_vec.unsqueeze(1), prediction_output=prediction_output)
                    .squeeze(1)
                    .squeeze(1)
                )
                # logging.warning(f"Joint: {joint_output.shape}")
                labels = joint_output.argmax(dim=1)

                blank_mask.bitwise_or_(labels == self.blank_index)
                # blank_mask_prev.bitwise_or_(blank_mask)

                if blank_mask.all():
                    all_blank_found = True
                else:
                    non_blank_indices = (blank_mask == False).nonzero(as_tuple=False).squeeze(1)
                    for i in non_blank_indices.tolist():
                        hyps[i].append(labels[i].item())
                    symbols_added += 1

                # logging.warning(f"Updating state {[sub_state.shape for sub_state in state]}")
                if blank_mask.any():
                    blank_indices = (blank_mask == True).nonzero(as_tuple=False).squeeze(1)
                    if prev_state is not None:
                        if isinstance(state, tuple):
                            for i, sub_state in enumerate(state):
                                sub_state[:, blank_indices] = prev_state[i][:, blank_indices]
                        else:
                            raise NotImplementedError
                    else:
                        if isinstance(state, tuple):
                            for i, sub_state in enumerate(state):
                                sub_state[:, blank_indices] = 0.0
                        else:
                            raise NotImplementedError

                    labels[blank_indices] = last_label[blank_indices]
                last_label = labels.clone()

        return hyps
