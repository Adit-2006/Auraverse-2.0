import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    """
    GRU-based temporal model for deepfake window classification.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, 512)

        Returns:
            Tensor of shape (B,) with fake probabilities
        """
        # GRU output: (B, T, hidden_dim)
        _, hidden = self.gru(x)

        # hidden: (num_layers, B, hidden_dim)
        last_hidden = hidden[-1]

        out = self.classifier(last_hidden)
        return out.squeeze(1)
