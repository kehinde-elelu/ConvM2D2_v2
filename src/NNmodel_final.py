import os
import torch, torchaudio

class PredictionHead(torch.nn.Module):
    """
    Ordinal-aware MOS predictor via classification over K bins (1..5) with Gaussian label softening.
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        # Equal-width bin centers from 1 to 5 inclusive
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, features):
        """
        features: (B, in_dim)
        Returns:
          logits: (B, K)
          probs:  (B, K)
          expected_mos: (B,)
        """
        logits = self.net(features)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.bins).sum(dim=-1)
        return logits, probs, expected

    @torch.no_grad()
    def build_soft_targets(self, mos, sigma=0.25):
        """
        mos: (B,) continuous scores in [1,5]
        Returns soft target distribution (B, K) using Gaussian kernel:
        y_k ∝ exp(-(s - c_k)^2 / (2 σ^2))
        """
        diff2 = (mos.unsqueeze(1) - self.bins.unsqueeze(0)) ** 2
        weights = torch.exp(-diff2 / (2 * sigma * sigma))
        return weights / weights.sum(dim=1, keepdim=True)

    def ordinal_loss(self, logits, soft_targets):
        """
        KL divergence between predicted log-probs and soft targets.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")

    def mse_monitor(self, expected_mos, target_mos):
        return torch.nn.functional.mse_loss(expected_mos, target_mos)


class TransformerPredictionHead(torch.nn.Module):
    """
    Single transformer layer (4 heads) + attention pooling -> 2-layer MLP over 20 ordinal bins.
    Accepts sequence features (B, T, D) or pooled (B, D).
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, n_heads=4, dropout=0.1, ff_multiplier=4):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=n_heads,
                dim_feedforward=hidden * ff_multiplier,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=1
        )
        # Attention pooling: learnable query vector
        self.query = torch.nn.Parameter(torch.randn(in_dim))
        self.attn_norm = torch.nn.LayerNorm(in_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, x):
        # x: (B, T, D) or (B, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)
        h = self.transformer(x)  # (B,T,D)
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,D)
        # scaled dot-product attention weights over time
        attn_logits = (h * q).sum(-1) / (h.size(-1) ** 0.5)  # (B,T)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B,T,1)
        pooled = (attn_weights * h).sum(1)  # (B,D)
        pooled = self.attn_norm(pooled)
        logits = self.mlp(pooled)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.bins).sum(dim=-1)
        return logits, probs, expected

    @torch.no_grad()
    def build_soft_targets(self, mos, sigma=0.25):
        diff2 = (mos.unsqueeze(1) - self.bins.unsqueeze(0)) ** 2
        weights = torch.exp(-diff2 / (2 * sigma * sigma))
        return weights / weights.sum(dim=1, keepdim=True)

    def ordinal_loss(self, logits, soft_targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")

    def mse_monitor(self, expected_mos, target_mos):
        return torch.nn.functional.mse_loss(expected_mos, target_mos)

