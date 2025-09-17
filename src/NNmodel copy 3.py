import os
import torch, torchaudio
import torch.nn as nn
import random
random.seed(1984)

# # Replace the failing import with robust fallbacks
# try:
#     from utmosv2.model import MultiSpecModelV2  # packaged path
#     print("*1"*50, MultiSpecModelV2, "*1"*50)
# except Exception:
#     try:
#         print("*2"*50, MultiSpecModelV2, "*2"*50)
#         from utmosv2.model._model import MultiSpecModelV2  # alt path in repo
#     except Exception:
#         MultiSpecModelV2 = None  # optional; only needed if activated

from utmosv2.model import MultiSpecModelV2
import importlib
cfg = importlib.import_module("utmosv2.config.fusion_stage3")
PRETRAINED_UTMOS_PATH = "/home/elelukeh/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth"


class PredictionHead(torch.nn.Module):
    """
    Ordinal-aware MOS predictor via classification over K bins (1..5) with Gaussian label softening.
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, dropout=0.3):
        super().__init__()
        self.num_bins = num_bins
        # Equal-width bin centers from 1 to 5 inclusive
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, features, spectrogram_x=None):

        # If features is (B, T, D), pool over T
        if features.dim() == 3:
            features_pooled = features.mean(dim=1)  # (B, D)
        else:
            features_pooled = features  # (B, D)

        logits = self.net(features_pooled)
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



# class PredictionHead_v2(torch.nn.Module):
#     """
#     Ordinal-aware MOS predictor via classification over K bins (1..5) with Gaussian label softening.
#     """
#     def __init__(self, in_dim=768, num_bins=20, hidden=256, dropout=0.3):
#         super().__init__()
#         self.num_bins = num_bins
#         # Equal-width bin centers from 1 to 5 inclusive
#         self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))
#         self.net = torch.nn.Sequential(
#             torch.nn.LayerNorm(in_dim),
#             torch.nn.Linear(in_dim, hidden),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout),
#             torch.nn.Linear(hidden, num_bins)
#         )
#         # Instantiate once so .to(device) moves it with the head
#         self.spec_encoder = MultiSpecModelV2(cfg)
        
#         self.spec_encoder.load_state_dict(torch.load(PRETRAINED_UTMOS_PATH))
        
#         for p in self.spec_encoder.parameters():
#             p.requires_grad = False
#         self.spec_encoder.eval()


#         if hasattr(self.spec_encoder, "fc"):
#             self.spec_encoder.fc = nn.Identity()

#     def forward(self, features, spectrogram_x):
#         # If features is (B, T, D), pool over T
#         if features.dim() == 3:
#             features_pooled = features.mean(dim=1)  # (B, D)
#         else:
#             features_pooled = features  # (B, D)

#         # Ensure spectrogram_x is on the same device/dtype as the spec encoder
#         enc_dev = next(self.spec_encoder.parameters()).device
#         enc_dtype = next(self.spec_encoder.parameters()).dtype
#         spectrogram_x = spectrogram_x.to(device=enc_dev, dtype=enc_dtype)

#         with torch.no_grad():
#             spec_features = self.spec_encoder(spectrogram_x)

#         # Optional: remove or keep for debugging
#         # print("*"*20, "shape of spec_features:", spec_features.shape, "*"*20)

#         logits = self.net(features_pooled)
#         probs = torch.softmax(logits, dim=-1)
#         expected = (probs * self.bins).sum(dim=-1)
#         return logits, probs, expected

#     @torch.no_grad()
#     def build_soft_targets(self, mos, sigma=0.25):
#         """
#         mos: (B,) continuous scores in [1,5]
#         Returns soft target distribution (B, K) using Gaussian kernel:
#         y_k ∝ exp(-(s - c_k)^2 / (2 σ^2))
#         """
#         diff2 = (mos.unsqueeze(1) - self.bins.unsqueeze(0)) ** 2
#         weights = torch.exp(-diff2 / (2 * sigma * sigma))
#         return weights / weights.sum(dim=1, keepdim=True)

#     def ordinal_loss(self, logits, soft_targets):
#         """
#         KL divergence between predicted log-probs and soft targets.
#         """
#         log_probs = torch.log_softmax(logits, dim=-1)
#         return torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")

#     def mse_monitor(self, expected_mos, target_mos):
#         return torch.nn.functional.mse_loss(expected_mos, target_mos)





class PredictionHead1(torch.nn.Module):
    """
    Ordinal-aware MOS predictor via classification over K bins (1..5) with Gaussian label softening.
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, dropout=0.3):
        super().__init__()
        self.num_bins = num_bins
        # Equal-width bin centers from 1 to 5 inclusive
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))

        in_dim = in_dim + 32 * 1251  
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, 20032), # 40064/2 = 20032
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(20032, 10016), #20032/2 = 10016
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(10016, 4096), #10016/2 = 4096
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4096, 2048), #4096/2 = 2048
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2048, 1024),  #2048/2 = 1024
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512),  #1024/2 = 512
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),  #512/2 = 256
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, num_bins)
        )

    def forward(self, features, spectrogram_x):

        # If features is (B, T, D), pool over T
        if features.dim() == 3:
            features_pooled = features.mean(dim=1)  # (B, D)
        else:
            features_pooled = features  # (B, D)
        
        if spectrogram_x.dim() > 2:
            spectrogram_x = spectrogram_x.flatten(start_dim=1)


        concat_feat = torch.cat([features_pooled, spectrogram_x], dim=1)

        logits = self.net(concat_feat)
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
    Input shapes:
      - x: (B, T, D)
        Example: torch.Size([32, 249, 768]) = 191,232 elements
      - spectrogram_x: (B, 64, 1251)
        Example: torch.Size([32, 64, 1251]) = 80,064 elements

    Before concatenation:
      - Flatten x to (B, T*D) if 3D, T*D = 249*768 = 191,232 ==> ([32, 191232])
      - Flatten spectrogram_x to (B, 64*1251) if 3D, 64*1251 = 80,064 ==> ([32, 80064])

    After flattening:
      - Both tensors are 2D and can be concatenated along dim=1
      - Resulting shape: (B, T*D + 64*1251) = (B, 271,296) ==> ([32, 271296])
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, n_heads=4, dropout=0.3, ff_multiplier=4):
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("bins", torch.linspace(1.0, 5.0, num_bins))

        mlp_input_dim = 249 * in_dim + 64 * 1251
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(mlp_input_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, x, spectrogram_x):
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        # Flatten spectrogram_x to (B, S)
        if spectrogram_x.dim() > 2:
            spectrogram_x = spectrogram_x.flatten(start_dim=1)
        # Both are now (B, *) and can be concatenated
        concat_feat = torch.cat([x, spectrogram_x], dim=1)  # (B, T*D + S)

        logits = self.mlp(concat_feat)
        probs = torch.softmax(logits, dim=-1)
        expected = (probs * self.bins).sum(dim=-1)
        return logits, probs, expected

    @torch.no_grad()
    def build_soft_targets(self, mos, sigma=0.25):
        diff2 = (mos.unsqueeze(1) - self.bins.unsqueeze(0)) ** 2
        weights = torch.exp(-diff2 / (2 * sigma * sigma))
        return weights / weights.sum(dim=1, keepdim=True)

    def ordinal_loss(self, logits, soft_targets):
        print("*"*20, logits.shape, "*"*20)
        print("="*20, soft_targets.shape, "="*20)
        log_probs = torch.log_softmax(logits, dim=-1)
        return torch.nn.functional.kl_div(log_probs, soft_targets, reduction="batchmean")

    def mse_monitor(self, expected_mos, target_mos):
        return torch.nn.functional.mse_loss(expected_mos, target_mos)



class TransformerPredictionHead_v3(torch.nn.Module):
    """
    Single transformer layer (4 heads) + attention pooling -> 2-layer MLP over 20 ordinal bins.
    Accepts sequence features (B, T, D) or pooled (B, D).
    """
    def __init__(self, in_dim=768, num_bins=20, hidden=256, n_heads=4, dropout=0.3, ff_multiplier=4):
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

        mlp_input_dim = in_dim + 64 * 1251
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(mlp_input_dim),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, num_bins)
        )

    def forward(self, x, spectrogram_x):
        # print("x"*50, "shape of x:", x.shape, "x"*50)
        # print("_x"*50, "shape of spectrogram_x:", spectrogram_x.shape, "_x"*50)
        # shape of x: torch.Size([32, 249, 768])
        # shape of spectrogram_x: torch.Size([32, 64, 1251])
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)
        h = self.transformer(x)  # (B,T,D)
        q = self.query.unsqueeze(0).unsqueeze(1)  # (1,1,D)
        attn_logits = (h * q).sum(-1) / (h.size(-1) ** 0.5)  # (B,T)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # (B,T,1)
        pooled = (attn_weights * h).sum(1)  # (B,D)
        pooled = self.attn_norm(pooled)

        # Concatenate spectrogram features and pooled features before passing into the mlp
        # Flatten spectrogram_x to (B, S)
        if spectrogram_x.dim() > 2:
            spectrogram_x = spectrogram_x.flatten(start_dim=1)
        concat_feat = torch.cat([pooled, spectrogram_x], dim=1)  # (B, D + S)

        logits = self.mlp(concat_feat)
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


class TransformerPredictionHead_v1(torch.nn.Module):
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
        attn_logits = (h * q).sum(-1)
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
