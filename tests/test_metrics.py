# This is the part of the Final Project
# "RNN-Transducer-based Losses for Speech Recognition on Noisy Targets"
# Originally published in https://github.com/artbataev/uol_final

import pytest

from min_rnnt.metrics import ExtendedWordErrorRate


class TestExtendedWordErrorRate:
    """Unit tests for ExtendedWordErrorRate metric based on jiwer"""

    def test_single(self):
        """Test single update"""
        metric = ExtendedWordErrorRate()
        metric.update(preds="one two three", target="one two tree")
        result = metric.compute()
        assert result["wer"].item() == pytest.approx(1 / 3)
        assert result["deletions"].item() == pytest.approx(0)
        assert result["substitutions"].item() == pytest.approx(1 / 3)
        assert result["insertions"].item() == pytest.approx(0)

    def test_multiple(self):
        """Test muptiple updates"""
        metric = ExtendedWordErrorRate()
        metric.update(preds=["one two three"], target=["one two tree"])  # one substitution
        metric.update(preds=["four five six"], target=["four five"])  # one insertions
        result = metric.compute()
        assert result["wer"].item() == pytest.approx(2 / 5)
        assert result["deletions"].item() == pytest.approx(0)
        assert result["substitutions"].item() == pytest.approx(1 / 5)
        assert result["insertions"].item() == pytest.approx(1 / 5)
