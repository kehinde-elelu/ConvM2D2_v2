# ...existing code...
import argparse
from typing import Tuple, List
# ...existing code...

def split_for_calibration(files: List[str], scores: List[float], calib_ratio=0.1):
    """Split off a calibration subset from training data."""
    n = len(files)
    c = max(1, int(n * calib_ratio))
    # Use last c samples for calibration (or shuffle beforehand for randomness)
    train_files, calib_files = files[:-c], files[-c:]
    train_scores, calib_scores = scores[:-c], scores[-c:]
    return (train_files, train_scores), (calib_files, calib_scores)

def build_loaders(train_files, train_scores, calib_files, calib_scores,
                  valid_files, valid_scores, loader, preproc,
                  batch_size_train=32, batch_size_eval=8, num_workers=4):
    trainset = AudioMOSDataset(train_files, train_scores, loader, preproc)
    calibset = AudioMOSDataset(calib_files, calib_scores, loader, preproc)
    validset = AudioMOSDataset(valid_files, valid_scores, loader, preproc)

    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    calibloader = DataLoader(calibset, batch_size=batch_size_eval, shuffle=False, num_workers=max(1, num_workers//2))
    validloader = DataLoader(validset, batch_size=batch_size_eval, shuffle=False, num_workers=max(1, num_workers//2))
    return trainloader, calibloader, validloader

def forward_feature(model, wave, device):
    with torch.no_grad():
        wav_embed = model.encode_clap_audio(wave)
        if wav_embed.dim() > 2:
            wav_embed = wav_embed.mean(dim=1)
    return wav_embed

def train_one_epoch(head, model, loader, optimizer, device, log_interval=10):
    head.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    for step, (wave, mos) in enumerate(loader, start=1):
        wave = wave.to(device)
        mos = mos.to(device)
        wav_embed = forward_feature(model, wave, device)
        logits, probs, pred_mos = head(wav_embed)
        soft_targets = head.build_soft_targets(mos, sigma=0.25)
        loss = head.ordinal_loss(logits, soft_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse = head.mse_monitor(pred_mos, mos).item()
        total_loss += loss.item()
        total_mse += mse
        n_batches += 1
        if step % log_interval == 0:
            print(f"Train Step {step} | OrdinalLoss {loss.item():.4f} | MSE {mse:.4f} | Pred mean {pred_mos.mean():.3f}")
    return total_loss / max(1, n_batches), total_mse / max(1, n_batches)

@torch.no_grad()
def eval_point(head, model, loader, device, desc="Eval"):
    head.eval()
    total_mse = 0.0
    n_batches = 0
    preds = []
    targets = []
    for wave, mos in loader:
        wave = wave.to(device)
        mos = mos.to(device)
        wav_embed = forward_feature(model, wave, device)
        _, _, pred_mos = head(wav_embed)
        mse = head.mse_monitor(pred_mos, mos).item()
        total_mse += mse
        n_batches += 1
        preds.append(pred_mos.cpu())
        targets.append(mos.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse_mean = total_mse / max(1, n_batches)
    print(f"{desc} | MSE {mse_mean:.4f}")
    return preds, targets, mse_mean

@torch.no_grad()
def conformal_calibrate(head, model, calib_loader, device, alpha=0.1):
    """
    Compute symmetric conformal interval half-width (q_hat) using absolute residuals.
    q_hat = quantile_{1-alpha}( |y - f(x)| )
    """
    preds, targets, _ = eval_point(head, model, calib_loader, device, desc="Calibration (point)")
    residuals = (targets - preds).abs()
    q_hat = torch.quantile(residuals, 1 - alpha).item()
    print(f"Conformal calibration | alpha={alpha:.3f} | q_hat={q_hat:.4f}")
    return q_hat

def apply_intervals(preds: torch.Tensor, q_hat: float, low=1.0, high=5.0):
    lower = (preds - q_hat).clamp(min=low, max=high)
    upper = (preds + q_hat).clamp(min=low, max=high)
    return lower, upper

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--calib_ratio", type=float, default=0.1, help="Fraction of training set for calibration")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level for conformal intervals")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight", type=str,
                    default="/egr/research-deeptech/elelukeh/MOS_project/M2D/m2d/models_m2d/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth")
    return ap.parse_args()

def main():
    args = parse_args()
    wav_dir = "data/main/DATA/wav"
    train_list, mos_trains_list = load_mos_txt("data/main/DATA/sets/train_mos_list.txt", wav_dir)
    valid_list, mos_valids_list = load_mos_txt("data/main/DATA/sets/val_mos_list.txt", wav_dir)

    print("Total Train samples:", len(train_list))
    print("Valid samples:", len(valid_list))

    # ----- Split calibration subset -----
    (train_files, train_scores), (calib_files, calib_scores) = split_for_calibration(train_list, mos_trains_list, calib_ratio=args.calib_ratio)
    print(f"Train subset: {len(train_files)} | Calibration subset: {len(calib_files)}")

    loader = AudioLoader()
    preproc = Preprocessor(T=16000 * 5)

    trainloader, calibloader, validloader = build_loaders(
        train_files, train_scores, calib_files, calib_scores,
        valid_list, mos_valids_list, loader, preproc,
        batch_size_train=args.batch_size
    )

    # --- Upstream model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UPSTREAM_OUT_DIM = 768
    model = PortableM2D(weight_file=args.weight, flat_features=True).to(device)

    head = PredictionHead(in_dim=UPSTREAM_OUT_DIM, num_bins=20).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    best_valid_mse = float("inf")
    q_hat_final = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_mse = train_one_epoch(head, model, trainloader, optimizer, device)
        print(f"Epoch {epoch} Train | Loss {train_loss:.4f} | MSE {train_mse:.4f}")

        # Point evaluation on validation
        _, _, valid_mse = eval_point(head, model, validloader, device, desc="Validation (point)")

        # Calibrate each epoch (optional) - could restrict to last epoch
        q_hat = conformal_calibrate(head, model, calibloader, device, alpha=args.alpha)
        q_hat_final = q_hat

        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            torch.save({
                "head_state": head.state_dict(),
                "q_hat": q_hat,
                "alpha": args.alpha
            }, "checkpoint_head.pt")
            print(f"Saved checkpoint (best MSE {best_valid_mse:.4f})")

    # Final interval demonstration on validation set
    preds, targets, _ = eval_point(head, model, validloader, device, desc="Validation (final point)")
    lower, upper = apply_intervals(preds, q_hat_final)
    coverage = ((targets >= lower) & (targets <= upper)).float().mean().item()
    print(f"Conformal intervals | alpha={args.alpha} | empirical coverage={coverage:.3f} | q_hat={q_hat_final:.4f}")
    print(f"Example intervals (first 5):")
    for i in range(min(5, len(preds))):
        print(f"y={targets[i]:.2f} pred={preds[i]:.2f} [{lower[i]:.2f}, {upper[i]:.2f}]")

if __name__ == "__main__":
    main()