# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

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
        """
        :param prediction_network: Prediction Network
        :param joint: Joint netowkr
        :param blank_index: index of the blank label
        :param max_symbols_per_step: maximum number of symbols per step (to prevent infinite looping)
        """
        super().__init__()
        self.prediction_network = prediction_network
        self.joint = joint
        self.blank_index = blank_index
        self.max_symbols_per_step = max_symbols_per_step

    def forward(
        self,
        encoder_output: torch.Tensor,  # BxDxT
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ):
        """
        Forward for training: compute Joint output from the combination of encoder and prediction network outputs
        """
        # get prediction network output using the target (with added <SOS> symbol)
        prediction_network_output, _ = self.prediction_network(
            input_prefix=targets, input_lengths=target_lengths, add_sos=True
        )
        # get Joint output
        joint_output = self.joint(
            encoder_output=encoder_output.transpose(1, 2), prediction_output=prediction_network_output
        )
        return joint_output

    def greedy_decode(
        self,
        encoder_output: torch.Tensor,  # [B, D, T]
        encoder_lengths: torch.Tensor,
    ) -> List[List[int]]:
        """
        Greedy decoding algorithm
        :param encoder_output: output from the encoder
        :param encoder_lengths: lengths of the outputs (for each encoded audio input)
        :return: List of hupotheses; each hypothesis - a list of token ids
        """
        # permute encoder output for easiness of usage
        encoder_output = encoder_output.permute(2, 0, 1)  # Time, Batch, NumFeatures
        max_time, batch_size, _ = encoder_output.shape
        # init hypotheses - empty arrays
        hyps = [[] for _ in range(batch_size)]

        # initial state is None (for LSTM)
        state = None
        device = encoder_output.device
        # we use <blank> symbols as <SOS>, fill the last label with <SOS> = <blank>
        last_label = torch.full([batch_size], fill_value=self.blank_index, dtype=torch.long, device=device)

        # loop over time frames
        for time_i in range(max_time):
            # get corresponding encoder vector
            encoder_vec = encoder_output[time_i]  # get current encoder vector

            # track added symbols to prevent infinite loop
            symbols_added = 0
            # mask for computing the output for batched decoding;
            # initially True if the sequence is not fully decoded
            blank_or_end_mask = time_i >= encoder_lengths  # use labels only if we are not out-of-sequence
            while not blank_or_end_mask.all() and (
                self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step
            ):
                prev_state = state  # cache previous state
                # compute prediction network with updated (last) labels
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

                # check if any labels are found
                if not blank_or_end_mask.all():
                    # some labels found, not all are blank
                    non_blank_indices = (~blank_or_end_mask).nonzero(as_tuple=False).squeeze(1)
                    # store found non-blank symbols
                    for i in non_blank_indices.tolist():
                        hyps[i].append(labels[i].item())
                    # increment added symbols
                    symbols_added += 1

                # if any of the output is blank
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
                            # reserved for future - other networks
                            raise NotImplementedError
                    else:
                        # prev_state is None, start of the decoding loop
                        if isinstance(state, tuple):
                            # restore 0 initial state for found blank labels
                            for i, sub_state in enumerate(state):
                                sub_state[:, blank_indices] = 0.0
                        else:
                            # reserved for future - other networks
                            raise NotImplementedError
                    # restore labels when the blank is found - we need only non-blank input to the Prediction network
                    labels[blank_indices] = last_label[blank_indices]
                last_label = labels.clone()
        # return decoded hypotheses
        return hyps
