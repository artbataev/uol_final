import torch
import torch.nn as nn
from nemo.collections.common.parts.rnn import LSTMDropout


class MinJoint(nn.Module):
    def __init__(
        self,
        encoder_output_dim: int,
        prediction_output_dim: int,
        joint_hidden_dim: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_output_dim, joint_hidden_dim)
        self.prediction_projection = nn.Linear(prediction_output_dim, joint_hidden_dim)
        self.joint_network = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(joint_hidden_dim, output_size),
        )

    def forward(self, encoder_output: torch.Tensor, prediction_output: torch.Tensor):
        encoder_projected = self.encoder_projection(encoder_output).unsqueeze(dim=2)  # B, T, 1, H
        prediction_projected = self.prediction_projection(prediction_output).unsqueeze(dim=1)  # B, 1, U, H
        return self.joint_network(encoder_projected + prediction_projected)


class MinPredictionNetwork(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, dropout: float = 0.2, num_layers: int = 1):
        super().__init__()
        self.blank_index = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=hidden_dim, padding_idx=self.blank_index)
        self.rnn = LSTMDropout(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            proj_size=hidden_dim,
            forget_gate_bias=1.0,
        )

    def forward(
        self, input_prefix: torch.Tensor, input_lengths: torch.Tensor, state=None, add_sos: bool = True
    ):
        batch_size = input_prefix.shape[0]
        if add_sos:
            input_prefix = torch.cat(
                [
                    torch.full(
                        (batch_size, 1),
                        fill_value=self.blank_index,
                        device=input_prefix.device,
                        dtype=input_prefix.dtype,
                    ),
                    input_prefix,
                ]
            )
        # TODO: packed sequence
        input_prefix_embed = self.embedding(input_prefix)
        output, state = self.rnn(input_prefix_embed, state)
        return output, state
