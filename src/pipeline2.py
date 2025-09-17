import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add project root

import torch, torchaudio
import warnings
from torch.utils.data import DataLoader, Sampler, Dataset
from typing import List
import argparse
import datetime
import torch.optim as optim
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import laion_clap
from transformers import Wav2Vec2Model
from utils import load_mos_txt, AudioLoader, Preprocessor, AudioMOSDataset, create_logger, forward_feature, export_learning_curves, apply_intervals
from NNmodel import PredictionHead_v3, TransformerPredictionHead
from preprocess import SimpleAugment, AugmentedDataset, split_for_calibration, build_loaders, StratifiedBatchSampler, extract_weighted_mel_spectrogram, batch_multi_spec_spectrograms_librosa

import random
random.seed(1984)

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

save_model_time = datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')

def train_one_epoch(head, model, loader, optimizer, device, log_interval=10, log=print, seq_mode=False):
    head.train()
    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0
    for step, (wave, mos) in enumerate(loader, start=1):
        wave = wave.to(device)
        mos = mos.to(device)
        wav_embed = forward_feature(model, wave, device, keep_sequence=seq_mode)

        spec_cfgs = [
            {'n_fft': 256, 'hop_length': 64, 'n_mels': 64},
            {'n_fft': 512, 'hop_length': 128, 'n_mels': 64},
            {'n_fft': 768, 'hop_length': 256, 'n_mels': 64},
            {'n_fft': 1024, 'hop_length': 512, 'n_mels': 64},
        ]
        weighted_mel = batch_multi_spec_spectrograms_librosa(wave, num_frames=4, spec_cfgs=spec_cfgs, device=device)

        # weighted_mel = batch_spectrograms(wave, num_frames=4, n_mels=64, n_fft=1024, hop_length=256, device=device)
        # weighted_mel = batch_spectrograms_librosa(wave, num_frames=4, n_mels=64, n_fft=1024, hop_length=256, sr=16000, device=None)
        # print("*"*20, 'weighted_mel shape: ' + str(weighted_mel.shape), "*"*20)
        # weighted_mel = extract_weighted_mel_spectrogram(
        #     wave,
        #     device,
        #     window_sizes=[256, 512, 768, 1024],
        #     hop_lengths=[64, 128, 256, 512],
        #     n_mels=32, weights=None
        # )

        logits, probs, pred_mos = head(wav_embed, spectrogram_x=weighted_mel)
        soft_targets = head.build_soft_targets(mos, sigma=0.25)
        kl_loss = head.ordinal_loss(logits, soft_targets)
        l1_loss = torch.nn.functional.l1_loss(pred_mos, mos)
        loss = kl_loss + 0.5 * l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mse = head.mse_monitor(pred_mos, mos).item()
        total_loss += loss.item()
        total_mse += mse
        n_batches += 1
        if step % log_interval == 0:
            log(f"Train Fold {step} | OrdinalLoss {loss.item():.4f} | MSE {mse:.4f} | Pred mean {pred_mos.mean():.3f}")
    return total_loss / max(1, n_batches), total_mse / max(1, n_batches)

@torch.no_grad()
def eval_point(head, model, loader, device, desc="Eval", log=print, seq_mode=False):
    head.eval()
    total_mse = 0.0
    total_loss = 0.0
    n_batches = 0
    preds = []
    targets = []
    VALSTEPS=0
    for wave, mos in loader:
        VALSTEPS+=1
        wave = wave.to(device)
        mos = mos.to(device)
        wav_embed = forward_feature(model, wave, device, keep_sequence=seq_mode)

        spec_cfgs = [
            {'n_fft': 256, 'hop_length': 64, 'n_mels': 64},
            {'n_fft': 512, 'hop_length': 128, 'n_mels': 64},
            {'n_fft': 768, 'hop_length': 256, 'n_mels': 64},
            {'n_fft': 1024, 'hop_length': 512, 'n_mels': 64},
        ]
        weighted_mel = batch_multi_spec_spectrograms_librosa(wave, num_frames=4, spec_cfgs=spec_cfgs, device=device)

        # weighted_mel = batch_spectrograms(wave, num_frames=4, n_mels=64, n_fft=1024, hop_length=256, device=device)
        # weighted_mel = batch_spectrograms_librosa(wave, num_frames=4, n_mels=64, n_fft=1024, hop_length=256, sr=16000, device=None)
        # print("*"*20, 'weighted_mel shape: ' + str(weighted_mel.shape), "*"*20)
        # weighted_mel = extract_weighted_mel_spectrogram(
        #     wave,
        #     device,
        #     window_sizes=[256, 512, 768, 1024],
        #     hop_lengths=[64, 128, 256, 512],
        #     n_mels=32, weights=None
        # )

        logits, probs, pred_mos = head(wav_embed, spectrogram_x=weighted_mel)
        soft_targets = head.build_soft_targets(mos, sigma=0.25)
        kl_loss = head.ordinal_loss(logits, soft_targets)
        l1_loss = torch.nn.functional.l1_loss(pred_mos, mos)
        loss = kl_loss + 0.5 * l1_loss

        mse = head.mse_monitor(pred_mos, mos).item()
        total_mse += mse
        total_loss += loss.item()
        n_batches += 1
        preds.append(pred_mos.cpu())
        targets.append(mos.cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse_mean = total_mse / max(1, n_batches)
    # loss_mean = total_loss / max(1, n_batches)
    loss_mean = total_loss / VALSTEPS
    log(f"{desc} | MSE {mse_mean:.4f} | Loss {loss_mean:.4f}")
    return preds, targets, mse_mean, loss_mean

@torch.no_grad()
def conformal_calibrate(head, model, calib_loader, device, alpha=0.1, log=print, seq_mode=False):
    """
    Compute symmetric conformal interval half-width (q_hat) using absolute residuals.
    q_hat = quantile_{1-alpha}( |y - f(x)| )
    """
    preds, targets, _, _ = eval_point(head, model, calib_loader, device, desc="Calibration (point)")
    residuals = (targets - preds).abs()
    q_hat = torch.quantile(residuals, 1 - alpha).item()
    log(f"Conformal calibration | alpha={alpha:.3f} | q_hat={q_hat:.4f}")
    return q_hat

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--calib_ratio", type=float, default=0.1, help="Fraction of training set for calibration")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level for conformal intervals")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight", type=str,
                    default="/egr/research-deeptech/elelukeh/MOS_project/M2D/m2d/models_m2d/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth")
    ap.add_argument("--out_dir", type=str, default="models", help="Directory to store checkpoints")
    ap.add_argument("--log_txt", type=str, default="logs", help="Optional path to append plain text training log")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs without improvement)")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="Minimum decrease in validation MSE to count as improvement")
    ap.add_argument("--use_transformer_head", action="store_true",
                    help="Use single-layer transformer + attention pooling head.")
    ap.add_argument("--stratified_batches", action="store_true",
                    help="Enable stratified MOS-balanced mini-batches for training.")
    ap.add_argument("--strat_bins", type=int, default=5, help="Number of MOS bins for stratified batching.")
    ap.add_argument("--use_augment", action="store_true",
                    help="Enable simple waveform augmentations (gain, noise, shift, stretch) for training.")
    ap.add_argument("--augment_p", type=float, default=0.5,
                    help="Per-augmentation application probability (shared).")
    ap.add_argument("--upstream", type=str, choices=["m2d", "wav2vec"], default="m2d",
                    help="Select upstream feature extractor.")
    ap.add_argument("--wav2vec_model", type=str, default="facebook/wav2vec2-base",
                    help="HF model id for wav2vec2 when --upstream wav2vec.")
    return ap.parse_args()



def main():
    args = parse_args()
    log, log_file = create_logger(args.log_txt)
    try:
        # 1. Data Preparation
        os.makedirs(args.out_dir, exist_ok=True)
        log(f"[INFO] Using device detection starting...")
        wav_dir = "data/main/DATA/wav"
        train_list, mos_trains_list = load_mos_txt("data/main/DATA/sets/train_mos_list.txt", wav_dir)
        valid_list, mos_valids_list = load_mos_txt("data/main/DATA/sets/val_mos_list.txt", wav_dir)
        test_list, mos_tests_list = load_mos_txt("data/main/DATA/sets/test_mos_list.txt", wav_dir)  # add a test file

        log("Total Train samples:", len(train_list))
        log("Total Valid samples:", len(valid_list))

        # ----- Split calibration subset -----
        (train_files, train_scores), (calib_files, calib_scores) = split_for_calibration(train_list, mos_trains_list, calib_ratio=args.calib_ratio)
        log(f"Train subset: {len(train_files)} | Calibration subset: {len(calib_files)}")

        loader = AudioLoader()
        preproc = Preprocessor(T=16000 * 5)

        augment = None
        if args.use_augment:
            augment = SimpleAugment(p=args.augment_p)
            log(f"[INFO] Data augmentation enabled (p={args.augment_p})")
            log("  - Gain ±{:.1f} dB | Noise SNR [{:.1f}, {:.1f}] dB".format(augment.max_gain_db, *augment.noise_snr_range))
            log(augment)

        trainloader, calibloader, validloader = build_loaders(
            train_files, train_scores, calib_files, calib_scores,
            valid_list, mos_valids_list, loader, preproc,
            batch_size_train=args.batch_size,
            stratified=args.stratified_batches,
            strat_bins=args.strat_bins,
            augment=augment
        )
        # Add test loader
        testset = AudioMOSDataset(test_list, mos_tests_list, loader, preproc)
        testloader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

        # 2. Model Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('DEVICE: ' + str(device))

        if args.upstream == "m2d":
            log(f"====[INFO] Loading M2D model from {args.weight}====")
            from m2d.examples.portable_m2d import PortableM2D
            model = PortableM2D(weight_file=args.weight, flat_features=True).to(device)
            UPSTREAM_OUT_DIM = 768
            log("====[INFO] Using M2D upstream.====")
        else:
            # Wav2Vec2 upstream
            log(f"====[INFO] Loading Wav2Vec2 model: {args.wav2vec_model}====")
            model = Wav2Vec2Model.from_pretrained(
                args.wav2vec_model, 
                cache_dir="/egr/research-deeptech/elelukeh/MOS_project/ConvM2D2V2/models/hf"
                ).to(device)
            UPSTREAM_OUT_DIM = model.config.hidden_size
            # log(f"====[INFO] Wav2Vec2 hidden size: {UPSTREAM_OUT_DIM}====")
            log(f"====[INFO] Using Wav2Vec2 upstream.====")

        head_cls = TransformerPredictionHead if args.use_transformer_head else PredictionHead_v3
        print('Using head class: ' + str(head_cls))
        head = head_cls(in_dim=UPSTREAM_OUT_DIM, num_bins=20).to(device)
        # optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer = optim.SGD(head.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)    
        # optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)


        log(f"[DEVICE] torch.cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"[DEVICE] current cuda index={torch.cuda.current_device()} name={torch.cuda.get_device_name()}")
        log(f"[DEVICE] Model first param device: {next(model.parameters()).device}")
        log(f"[DEVICE] Head first param device: {next(head.parameters()).device}")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        best_valid_loss = float("inf")
        # Track metrics for plotting
        metrics = {"train_mse": [], "valid_mse": [], "train_loss": [], "valid_loss": [], "q_hat": []}

        # 3. Training Loop
        for epoch in range(1, args.epochs + 1):
            log(f"\n=== Epoch {epoch}/{args.epochs} ===")
            train_loss, train_mse = train_one_epoch(
                head, model, trainloader, optimizer, device, log=log, seq_mode=args.use_transformer_head
            )
            log(f"Epoch {epoch} Train | Loss {train_loss:.4f} | MSE {train_mse:.4f}")

            _, _, valid_mse, valid_loss = eval_point(
                head, model, validloader, device, desc="Validation (point)", log=log, seq_mode=args.use_transformer_head
            )
            q_hat = conformal_calibrate(
                head, model, calibloader, device, alpha=args.alpha, log=log, seq_mode=args.use_transformer_head
            )
            q_hat_final = q_hat

            # Save metrics
            metrics["train_mse"].append(train_mse)
            metrics["valid_mse"].append(valid_mse)
            metrics["train_loss"].append(train_loss)
            metrics["valid_loss"].append(valid_loss)
            metrics["q_hat"].append(q_hat)

            improvement = (best_valid_loss - valid_loss) > args.min_delta
            if improvement:
                best_valid_loss = valid_loss
                patience_ctr = 0
                ckpt_path = os.path.join(args.out_dir, f"checkpoint_head_{args.upstream}_{save_model_time}.pt")
                torch.save({
                    "head_state": head.state_dict(),
                    "q_hat": q_hat,
                    "alpha": args.alpha,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss
                }, ckpt_path)
                log(f"Saved checkpoint (best Loss {best_valid_loss:.4f})")
                log(f"  - {ckpt_path}")
                last_saved_epoch = epoch
            else:
                patience_ctr += 1
                log(f"No improvement (Δ={best_valid_loss - valid_loss:.6f}); patience {patience_ctr}/{args.patience}")
                if patience_ctr >= args.patience:
                    log(f"Early stopping triggered at epoch {epoch}")
                    early_stopped = True
                    break

        # Export metrics and plot learning curves
        if metrics["train_mse"]:
            _out = export_learning_curves(
                metrics=metrics,
                out_dir=args.out_dir,
                upstream=args.upstream,
                last_saved_epoch=last_saved_epoch,
                log=log
            )

        if early_stopped:
            log(f"Early stopped. Best checkpoint was from epoch {last_saved_epoch}. Reloading for test...")
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_head_{args.upstream}_{save_model_time}.pt")
            if os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                head.load_state_dict(ckpt["head_state"])
                q_hat_final = ckpt.get("q_hat", q_hat_final)

        preds, targets, _, _ = eval_point(
            head, model, testloader, device, desc="Test (point)", log=log, seq_mode=args.use_transformer_head
        )
        lower, upper = apply_intervals(preds, q_hat_final)
        coverage = ((targets >= lower) & (targets <= upper)).float().mean().item()
        log(f"[TEST] Conformal coverage={coverage:.3f} (alpha={args.alpha}, q_hat={q_hat_final:.4f})")
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()

# python src/pipeline2.py --use_transformer_head
# python src/pipeline2.py --use_transformer_head --stratified_batches --strat_bins 5 --use_augment (nope dont use augment)


# python src/pipeline2.py --use_transformer_head --stratified_batches --strat_bins 5 --upstream wav2vec
# python src/pipeline2.py --use_transformer_head --stratified_batches --strat_bins 5 --upstream m2d 
# 
# python src/pipeline2.py --use_transformer_head --stratified_batches --strat_bins 5 --upstream wav2vec --wav2vec_model facebook/wav2vec2-large
# python src/pipeline2.py --use_transformer_head --stratified_batches --strat_bins 5 --upstream m2d --wav2vec_model facebook/wav2vec2-large-960h

'''
1. CUDA_VISIBLE_DEVICES=2 python src/pipeline2.py --upstream m2d
2. python src/pipeline2.py --upstream wav2vec --stratified_batches --strat_bins 5


two runnnng 
1. with stratified batches, SGD
2. without stratified batches,  AdamW


without stratified batches, SGD, increasing the nn layers 
'''



'''
CUDA_VISIBLE_DEVICES=0 python src/pipeline2.py --upstream wav2vec


now doing =====>
CUDA_VISIBLE_DEVICES=1 python src/pipeline2.py --upstream m2d


CUDA_VISIBLE_DEVICES=1 ./predict.sh wav2vec


Start a new session: tmux
Start with a name: tmux new -s mysession
Detach (leave it running): Ctrl+b then d
List sessions: tmux ls
Attach to existing: 
    tmux attach -t mysession
    tmux attach -t mysession1
Create if absent (attach or new): tmux new -As mysession
Kill a session: tmux kill-session -t mysession
Common splits: Ctrl+b then % (vertical split) Ctrl+b then " (horizontal split)
Switch panes: Ctrl+b then arrow key
Exit all panes (ends session) or detach to keep running.

Ideas:
    Datasets:
    1. Stratify batches by MOS to avoid mini-batch score drift. (DONE)
    2. Add simple augmentations: small gain, mild noise, time-shift, time-stretch (avoid heavy distortions that change quality). (DONE via --use_augment)
    3. Clip or winsorize extreme MOS if rare
    4. Try different upstream models (e.g., Wav2CLIP, HuBERT, WavLM, etc.)
    
    Model:

    https://towardsdatascience.com/uncertainty-quantification-and-why-you-should-care-3f8a651f1956/
'''