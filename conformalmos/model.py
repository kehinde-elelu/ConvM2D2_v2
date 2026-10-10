"""Baseline head adapted from src/NNmodel.py without optional UTMOS imports."""
import torch

class PredictionHead(torch.nn.Module):
    def __init__(self, in_dim=768, num_bins=20, hidden=256, dropout=0.3):
        super().__init__()
        self.register_buffer("bins", torch.linspace(1., 5., num_bins))
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden, num_bins))

    def forward(self, features):
        if features.ndim == 3:
            features = features.mean(dim=1)
        probs = self.net(features).softmax(dim=-1)
        return (probs * self.bins).sum(dim=-1)
