import k2
import pytest
import torch
from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss

from min_rnnt.losses import GraphBypassTransducerLoss


class TestBypassTransducerLoss:
    def test_grid_vs_compose_equivalence(self):
        vocab_size = 10
        units_tensor = torch.tensor([2, 5, 1, 6, 0])
        loss_composed = GraphBypassTransducerLoss(blank=vocab_size - 1, use_grid_implementation=False)
        loss_grid = GraphBypassTransducerLoss(blank=vocab_size - 1, use_grid_implementation=True)

        composed_lattice = loss_composed.get_composed_lattice(
            units_tensor=units_tensor, num_frames=10, vocab_size=vocab_size
        )
        composed_lattice = k2.connect(composed_lattice)

        grid_lattice = loss_grid.get_grid(units_tensor=units_tensor, num_frames=10, vocab_size=vocab_size)

        assert k2.is_rand_equivalent(composed_lattice, grid_lattice, log_semiring=True)

    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    @pytest.mark.parametrize("skip_token_mode", ["sumexcl", "constant", "mean", "max", "maxexcl", "meanexcl"])
    def test_match_rnnt_inf_penalty(self, use_grid_implementation: bool, skip_token_mode: str):
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        bypasst_loss = GraphBypassTransducerLoss(
            blank=blank_id,
            skip_token_penalty=float("-inf"),
            skip_token_mode=skip_token_mode,
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
        )
        rnnt_loss = GraphRnntLoss(blank=blank_id, double_scores=True)

        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        logits2 = logits.detach()
        logits2.requires_grad_(True)

        rnnt_loss_value = rnnt_loss(logits, targets, input_lengths, target_lengths)
        rnnt_loss_value.mean().backward()

        bypasst_loss_value = bypasst_loss(logits2, targets, input_lengths, target_lengths)
        bypasst_loss_value.mean().backward()

        assert torch.allclose(rnnt_loss_value, bypasst_loss_value)
        assert torch.allclose(logits.grad, logits2.grad)
