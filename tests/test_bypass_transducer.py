# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

import k2
import pytest
import torch
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss

from min_rnnt.losses import GraphBypassTransducerLoss


class TestBypassTransducerLoss:
    """
    Bypass-Transducer loss unit tests.
    We test that
    - grid and composed lattices are equivalent
    - when skip token penalty is -inf, the loss and gradient are equivalent to RNN-T
    (compare with the reference loss - GraphRnntLoss from NeMo)
    """

    def test_grid_vs_compose_equivalence(self):
        """Test that grid and composed lattices are equivalent"""
        # use small vocab of 10 and a tensor with units [2, 5, 1, 6, 0]
        vocab_size = 10
        units_tensor = torch.tensor([2, 5, 1, 6, 0])
        loss_composed = GraphBypassTransducerLoss(blank=vocab_size - 1, use_grid_implementation=False)
        loss_grid = GraphBypassTransducerLoss(blank=vocab_size - 1, use_grid_implementation=True)

        # get composed lattice
        composed_lattice = loss_composed.get_composed_lattice(
            units_tensor=units_tensor, num_frames=10, vocab_size=vocab_size
        )
        # apply "connect" operation to remove unreachable states (which do not affect loss computation)
        composed_lattice = k2.connect(composed_lattice)

        # get grid (directly constructed lattice)
        grid_lattice = loss_grid.get_grid(units_tensor=units_tensor, num_frames=10, vocab_size=vocab_size)

        # check equivalence
        assert k2.is_rand_equivalent(composed_lattice, grid_lattice, log_semiring=True)

    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    @pytest.mark.parametrize("skip_token_mode", ["sumexcl", "constant", "mean", "max", "maxexcl"])
    def test_match_rnnt_inf_penalty(self, use_grid_implementation: bool, skip_token_mode: str):
        """When skip token penalty is -inf, the loss and gradient should be equivalent to RNN-T"""
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        # Bypass-Transducer loss
        bypasst_loss = GraphBypassTransducerLoss(
            blank=blank_id,
            skip_token_penalty=float("-inf"),  # -inf skip token penalty - for equivalence to RNN-T
            skip_token_mode=skip_token_mode,
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
        )
        # etalon: RNN-T loss
        rnnt_loss = GraphRnntLoss(blank=blank_id, double_scores=True)

        # generate random Joint output and targets
        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        # clone Joint output (logits) to get gradient from the second computation
        logits2 = logits.detach()
        logits2.requires_grad_(True)

        # compute RNN-T loss
        rnnt_loss_value = rnnt_loss(logits, targets, input_lengths, target_lengths)
        # backward for gradients
        rnnt_loss_value.mean().backward()

        # compute Bypass-Transducer loss
        bypasst_loss_value = bypasst_loss(logits2, targets, input_lengths, target_lengths)
        # backward for gradients
        bypasst_loss_value.mean().backward()

        # test loss values are the same
        assert torch.allclose(rnnt_loss_value, bypasst_loss_value)
        # test gradients are the same
        assert torch.allclose(logits.grad, logits2.grad)
