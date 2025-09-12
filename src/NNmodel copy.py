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


class TransformerPredictionHead1(torch.nn.Module):
    """
    Two modes:
      1) CRNN style (use_crnn_style=True): Conv1d stack -> global mean -> regression MLP (produces scalar MOS).
      2) Hybrid (use_crnn_style=False): (optional conv stack) -> Transformer + attention pooling -> ordinal bins.
    NOTE: In CRNN style we return (logits, probs, expected) for interface compatibility:
          logits = raw scalar prediction (B,1)
          probs  = sigmoid(logits) (B,1)   (placeholder, not ordinal distribution)
          expected = logits.squeeze(-1)
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, n_heads=4, dropout=0.1,
                 ff_multiplier=4, use_conv_blocks=True, use_crnn_style=True):
        super().__init__()
        self.num_bins = num_bins
        self.use_conv_blocks = use_conv_blocks
        self.use_crnn_style = use_crnn_style
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))

        print(f"TransformerPredictionHead1 config | conv_blocks={use_conv_blocks} | crnn_style={use_crnn_style}")

        if use_conv_blocks:
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
            if not self.use_crnn_style:
                if 768 != in_dim:
                    self.post_conv_proj = torch.nn.Linear(768, in_dim)
                else:
                    self.post_conv_proj = torch.nn.Identity()
            else:
                self.post_conv_proj = torch.nn.Identity()
        else:
            self.post_conv_proj = torch.nn.Identity()

        if not self.use_crnn_style:
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
            self.query = torch.nn.Parameter(torch.randn(in_dim))
            self.attn_norm = torch.nn.LayerNorm(in_dim)
            # Ordinal MLP head (classification over bins)
            self.ordinal_mlp = torch.nn.Sequential(
                torch.nn.LayerNorm(in_dim),
                torch.nn.Linear(in_dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, num_bins)
            )
        else:
            self.transformer = None
            self.query = None
            self.attn_norm = torch.nn.Identity()
            # Regression MLP (CRNN style) as requested
            self.overall_mlp_layer1 = torch.nn.Linear(in_features=768, out_features=256)
            self.overall_mlp_layer2 = torch.nn.Linear(in_features=256, out_features=64)
            self.overall_mlp_layer3 = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)

        if self.use_conv_blocks:
            MIN_T = 16
            if x.size(1) < MIN_T:
                repeat_factor = max(1, MIN_T // x.size(1))
                x = x.repeat(1, repeat_factor, 1)
            x_c = x.transpose(1, 2)          # (B,D,T)
            x_c = self.conv_block1(x_c)
            x_c = self.conv_block2(x_c)
            x_c = self.conv_block3(x_c)
            x_c = self.conv_block4(x_c)      # (B,768,T')
            if self.use_crnn_style:
                feat = x_c.mean(dim=2)       # (B,768)
            else:
                x = x_c.transpose(1, 2)      # (B,T',768)
                x = self.post_conv_proj(x)   # (B,T',in_dim)
        else:
            if self.use_crnn_style:
                feat = x.mean(dim=1)         # (B,D)

        if self.use_crnn_style:
            h1 = torch.relu(self.overall_mlp_layer1(feat))
            h2 = torch.relu(self.overall_mlp_layer2(h1))
            mos = self.overall_mlp_layer3(h2)        # (B,1)
            logits = mos                              # treat as logits placeholder
            probs = torch.sigmoid(logits)             # (B,1) dummy prob
            expected = mos.squeeze(-1)                # (B,)
            return logits, probs, expected
        else:
            h = self.transformer(x)                   # (B,T*,D)
            q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,D)
            attn_logits = (h * q).sum(-1) / (h.size(-1) ** 0.5)
            attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
            pooled = (attn_weights * h).sum(1)
            pooled = self.attn_norm(pooled)
            logits = self.ordinal_mlp(pooled)         # (B,num_bins)
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