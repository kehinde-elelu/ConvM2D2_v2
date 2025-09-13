import os
import torch, torchaudio

import random
random.seed(1984)


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




class TransformerPredictionHead1(torch.nn.Module):
    """
    CRNN-style audio-only ordinal MOS predictor.
    4 Conv1d blocks + MLP -> logits over num_bins spanning [1,5].
    Accepts (B, D) or (B, T, D). If (B, D), repeats along time before Conv1d.
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, repeat_len=16, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.num_bins = num_bins
        self.repeat_len = repeat_len
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))

        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=256, out_channels=768, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2)
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Linear(768, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, x):
        # x: (B, D) or (B, T, D)
        # print("x"*50, x.shape, "x"*50)
        b, t, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected feature dim {self.in_dim}, got {d}")
        x = x.transpose(1, 2)  # (B, D, T)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.mean(dim=2)  # (B, 768)

        logits = self.mlp(x)                  # (B, K)
        # print("x"*50, logits.shape, "x"*50)
        probs = torch.softmax(logits, dim=-1) # (B, K)
        expected = (probs * self.bins).sum(dim=-1)  # (B,)
        return logits, probs, expected

    @torch.no_grad()
    def build_soft_targets(self, mos, sigma=0.25):
        # mos: (B,)
        diff2 = (mos.unsqueeze(1) - self.bins.unsqueeze(0)) ** 2
        weights = torch.exp(-diff2 / (2 * sigma * sigma))
        return weights / weights.sum(dim=1, keepdim=True)

    def ordinal_loss(self, logits, soft_targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")

    def mse_monitor(self, expected_mos, target_mos):
        return torch.nn.functional.mse_loss(expected_mos, target_mos)
