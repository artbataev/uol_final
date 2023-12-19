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
        encoder_output: torch.Tensor,
        encoder_lengths: torch.Tensor,
    ) -> List[List[int]]:
        # encoder_output: [B, T, D]
        batch_size = encoder_output.shape[0]
        hyps = [[] for _ in range(batch_size)]

        state = None
        device = encoder_output.device
        last_label = torch.full([batch_size], fill_value=self.blank_index, dtype=torch.long, device=device)
        # blank_mask = torch.full([batch_size], fill_value=False, dtype=torch.bool, device=device)

        for time_i in range(encoder_output.shape[1]):
            encoder_vec = encoder_output[:, time_i]

            all_blank_found = False
            symbols_added = 0
            blank_mask = time_i >= encoder_lengths
            while not all_blank_found and (self.max_symbols is None or symbols_added < self.max_symbols):
                blank_mask_prev = blank_mask.clone()
                prev_state = state
                prediction_output, state = self.prediction_network(input_prefix=last_label, state=state, add_sos=False)

                logp = (
                    self.joint(encoder_output=encoder_vec, prediction_output=prediction_output).squeeze(1).squeeze(1)
                )
                _, labels = logp.max(1)

                blank_mask.bitwise_or_(labels == self.blank_index)
                blank_mask_prev.bitwise_or_(blank_mask)

                if blank_mask.all():
                    all_blank_found = True
                else:
                    blank_indices = (blank_mask == 1).nonzero(as_tuple=False)
                    # TODO: check first symbol
                    if prev_state is not None:
                        if isinstance(state, tuple):
                            for i, sub_state in enumerate(state):
                                sub_state[blank_indices] = prev_state[i][blank_indices]
                        else:
                            raise NotImplementedError

                    labels[blank_indices] = last_label[blank_indices]
                    last_label = labels.clone()

                    for batch_i, label in enumerate(labels.tolist()):
                        hyps[batch_i].append(label)
                    symbols_added += 1
        return hyps
