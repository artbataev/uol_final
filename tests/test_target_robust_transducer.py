import k2
import pytest
import torch

from min_rnnt.losses import GraphBypassTransducerLoss, GraphStarTransducerLoss, GraphTargetRobustTransducerLoss


class TestTargetRobustTransducerLoss:
    """
    Target-Robust Transducer loss unit tests.
    We test that
    - grid and composed lattices are equivalent
    - when skip token penalty is -inf, the loss and gradient are equivalent to Star-Transducer loss
    - when skip frame penalty is -inf, the loss and gradient are equivalent to Bypass-Transducer loss
    """

    def test_grid_vs_compose_equivalence(self):
        """Test that grid and composed lattices are equivalent"""
        # use small vocab of 10 and a tensor with units [2, 5, 1, 6, 0]
        vocab_size = 10
        units_tensor = torch.tensor([2, 5, 1, 6, 0])
        loss_composed = GraphTargetRobustTransducerLoss(blank=vocab_size - 1, use_grid_implementation=False)
        loss_grid = GraphTargetRobustTransducerLoss(blank=vocab_size - 1, use_grid_implementation=True)

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

    @pytest.mark.parametrize("skip_token_penalty", [0.0, -1.0, -5.0])
    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    @pytest.mark.parametrize("skip_token_mode", ["sumexcl", "constant", "mean", "max", "maxexcl", "meanexcl"])
    def test_match_bypass_transducer(
        self, skip_token_penalty: float, use_grid_implementation: bool, skip_token_mode: str
    ):
        """When skip frame penalty is -inf, the loss and gradient should be equivalent to Bypass-Transducer loss"""
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        trt_loss = GraphTargetRobustTransducerLoss(
            blank=blank_id,
            skip_frame_penalty=float("-inf"),  # skip frame penalty -inf
            skip_token_penalty=skip_token_penalty,
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
            skip_token_mode=skip_token_mode,
        )
        # etalon: Bypass-Transducer loss
        bypass_loss = GraphBypassTransducerLoss(
            blank=blank_id, skip_token_penalty=skip_token_penalty, double_scores=True, skip_token_mode=skip_token_mode
        )

        # generate random logits (outputs from Joint) and targets
        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        # clone logits to get the gradient from the second computation
        logits2 = logits.detach()
        logits2.requires_grad_(True)

        # compute Target Robust transducer loss values
        trt_loss_value = trt_loss(logits, targets, input_lengths, target_lengths)
        # backward for gradient
        trt_loss_value.mean().backward()

        # compute Bypass Transducer loss values
        bypass_loss_value = bypass_loss(logits2, targets, input_lengths, target_lengths)
        # backward for gradients
        bypass_loss_value.mean().backward()

        # check values are the same
        assert torch.allclose(trt_loss_value, bypass_loss_value)
        # check gradients are the same
        assert torch.allclose(logits.grad, logits2.grad)

    @pytest.mark.parametrize("skip_frame_penalty", [0.0, -1.0, -5.0])
    @pytest.mark.parametrize("use_grid_implementation", [False, True])
    def test_match_star_transducer(self, skip_frame_penalty: float, use_grid_implementation: bool):
        """When skip token penalty is -inf, the loss and gradient should be equivalent to Star-Transducer loss"""
        vocab_size = 10
        blank_id = vocab_size - 1
        encoder_output_length = 7
        units_lengths = 5
        device = torch.device("cpu")
        # instantiate Target Robust Transducer loss
        trt_loss = GraphTargetRobustTransducerLoss(
            blank=blank_id,
            skip_frame_penalty=skip_frame_penalty,
            skip_token_penalty=float("-inf"),  # skip token penalty -inf
            double_scores=True,
            use_grid_implementation=use_grid_implementation,
            skip_token_mode="constant",
        )
        # etalon: Star-Transducer loss
        star_loss = GraphStarTransducerLoss(blank=blank_id, skip_frame_penalty=skip_frame_penalty, double_scores=True)

        # generate random logits (Joint output) and targets
        logits = torch.rand(
            [2, encoder_output_length, units_lengths + 1, vocab_size], requires_grad=True, device=device
        )
        targets = torch.randint(0, vocab_size - 1, [2, units_lengths], device=device)
        input_lengths = torch.tensor([encoder_output_length, encoder_output_length - 1]).to(device)
        target_lengths = torch.tensor([units_lengths, units_lengths - 2]).to(device)

        # clone logits for the gradient from the second computation
        logits2 = logits.detach()
        logits2.requires_grad_(True)

        # compute Target-Robust Transducer loss values
        trt_loss_value = trt_loss(logits, targets, input_lengths, target_lengths)
        # backward for gradient
        trt_loss_value.mean().backward()

        # compute etalon - Star Transducer loss values
        star_loss_value = star_loss(logits2, targets, input_lengths, target_lengths)
        # backward for gradient
        star_loss_value.mean().backward()

        # check values are the same
        assert torch.allclose(trt_loss_value, star_loss_value)
        # check gradients are the same
        assert torch.allclose(logits.grad, logits2.grad)
