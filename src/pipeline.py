import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add project root

import torch
import warnings
from torch.utils.data import DataLoader
from utils import load_mos_txt, AudioLoader, Preprocessor, AudioMOSDataset, PredictionHead

# Optional: add external M2D repo root if different from this project
EXTERNAL_M2D_ROOT = "/egr/research-deeptech/elelukeh/MOS_project/M2D"
if os.path.isdir(os.path.join(EXTERNAL_M2D_ROOT, "m2d")) and EXTERNAL_M2D_ROOT not in sys.path:
    sys.path.insert(0, EXTERNAL_M2D_ROOT)

try:
    from m2d.examples.portable_m2d import PortableM2D
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Cannot import m2d. Ensure the M2D repo exists and has an __init__.py or run: pip install -e /egr/research-deeptech/elelukeh/MOS_project/M2D"
    ) from e


def main():
    wav_dir = "data/main/DATA/wav"
    train_list, mos_trains_list = load_mos_txt("data/main/DATA/sets/train_mos_list.txt", wav_dir)
    valid_list, mos_valids_list = load_mos_txt("data/main/DATA/sets/val_mos_list.txt", wav_dir)

    print("Train samples:", len(train_list))
    print("Valid samples:", len(valid_list))

    loader = AudioLoader()
    preproc = Preprocessor(T=16000 * 5)  # 5 seconds

    trainset = AudioMOSDataset(train_list, mos_trains_list, loader, preproc)
    validset = AudioMOSDataset(valid_list, mos_valids_list, loader, preproc)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=8, shuffle=False, num_workers=2)

    # --- M2D2CLAPEncoder setup (placeholder) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UPSTREAM_MODEL = 'M2D2CLAP'
    if UPSTREAM_MODEL == 'M2D2CLAP':
        UPSTREAM_OUT_DIM = 768
        weight = '/egr/research-deeptech/elelukeh/MOS_project/M2D/m2d/models_m2d/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth'
        model = PortableM2D(weight_file=weight, flat_features=True).to(device)
    else:
        print(f'*** ERROR *** Model type {UPSTREAM_MODEL} not supported.')
        return

    head = PredictionHead(in_dim=UPSTREAM_OUT_DIM, num_bins=20).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)

    model.eval()          # feature extractor frozen (optional)
    for p in model.parameters():
        p.requires_grad = False

    head.train()
    for step, (wave, mos) in enumerate(trainloader, start=1):
        wave = wave.to(device)          # (B, T)
        mos = mos.to(device)            # (B,)
        with torch.no_grad():
            wav_embed = model.encode_clap_audio(wave)  # (B, 768) or (B, T', 768)
            if wav_embed.dim() > 2:
                wav_embed = wav_embed.mean(dim=1)

        logits, probs, pred_mos = head(wav_embed)
        soft_targets = head.build_soft_targets(mos, sigma=0.25)
        loss = head.ordinal_loss(logits, soft_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            mse = head.mse_monitor(pred_mos, mos).item()
            print(f"Step {step} | OrdinalLoss {loss.item():.4f} | MSE {mse:.4f} | Pred mean {pred_mos.mean():.3f}")


if __name__ == "__main__":
    main()