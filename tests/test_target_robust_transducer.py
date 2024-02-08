import pytest
import torch

from min_rnnt.losses import GraphBypassTransducerLoss, GraphStarTransducerLoss, GraphTargetRobustTransducerLoss


class TestTargetRobustTransducerLoss:
    @pytest.mark.parametrize("skip_token_penalty", [0.0, -1.0, -5.0])
    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    @pytest.mark.parametrize("skip_token_mode", ["constant", "mean", "max", "maxexcl", "meanexcl"])
    def test_match_bypass_transducer(
        self, skip_token_penalty: float, use_grid_implementation: bool, skip_token_mode: str
    ):
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        trt_loss = GraphTargetRobustTransducerLoss(
            blank=blank_id,
            skip_frame_penalty=float("-inf"),
            skip_token_penalty=skip_token_penalty,
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
            skip_token_mode=skip_token_mode,
        )
        bypass_loss = GraphBypassTransducerLoss(
            blank=blank_id, skip_token_penalty=skip_token_penalty, double_scores=True, skip_token_mode=skip_token_mode
        )

        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        logits2 = logits.detach()
        logits2.requires_grad_(True)

        trt_loss_value = trt_loss(logits, targets, input_lengths, target_lengths)
        trt_loss_value.mean().backward()

        bypass_loss_value = bypass_loss(logits2, targets, input_lengths, target_lengths)
        bypass_loss_value.mean().backward()

        assert torch.allclose(trt_loss_value, bypass_loss_value)
        assert torch.allclose(logits.grad, logits2.grad)

    @pytest.mark.parametrize("skip_frame_penalty", [0.0, -1.0, -5.0])
    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    def test_match_star_transducer(self, skip_frame_penalty: float, use_grid_implementation: bool):
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        trt_loss = GraphTargetRobustTransducerLoss(
            blank=blank_id,
            skip_frame_penalty=skip_frame_penalty,
            skip_token_penalty=float("-inf"),
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
            skip_token_mode="constant",
        )
        star_loss = GraphStarTransducerLoss(blank=blank_id, skip_frame_penalty=skip_frame_penalty, double_scores=True)

        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        logits2 = logits.detach()
        logits2.requires_grad_(True)

        trt_loss_value = trt_loss(logits, targets, input_lengths, target_lengths)
        trt_loss_value.mean().backward()

        star_loss_value = star_loss(logits2, targets, input_lengths, target_lengths)
        star_loss_value.mean().backward()

        assert torch.allclose(trt_loss_value, star_loss_value)
        assert torch.allclose(logits.grad, logits2.grad)
