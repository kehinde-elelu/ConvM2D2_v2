import sys
import os
import argparse
import json
import numpy as np
import torch
import scipy.stats
from torch.utils.data import DataLoader

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_mos_txt, AudioLoader, Preprocessor, AudioMOSDataset, PredictionHead

EXTERNAL_M2D_ROOT = "/egr/research-deeptech/elelukeh/MOS_project/M2D"
if os.path.isdir(os.path.join(EXTERNAL_M2D_ROOT, "m2d")) and EXTERNAL_M2D_ROOT not in sys.path:
    sys.path.insert(0, EXTERNAL_M2D_ROOT)

try:
    from m2d.examples.portable_m2d import PortableM2D
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Cannot import m2d. Install / point EXTERNAL_M2D_ROOT correctly (pip install -e M2D)."
    ) from e


def parse_args():
    ap = argparse.ArgumentParser(description="Prediction & evaluation script for MOS model (pipeline2).")
    ap.add_argument("--test_list", type=str, default="data/main/DATA/sets/test_mos_list.txt",
                    help="Text file with '<rel_path> <score>' per line.")
    ap.add_argument("--wav_dir", type=str, default="data/main/DATA/wav", help="Base directory for wav files.")
    ap.add_argument("--checkpoint", type=str, default="models/checkpoint_head.pt",
                    help="Checkpoint produced by pipeline2.py (contains head_state & q_hat).")
    ap.add_argument("--weight",
                    default="/egr/research-deeptech/elelukeh/MOS_project/M2D/m2d/models_m2d/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth",
                    help="Upstream M2D weight file.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_csv", type=str, default="result/predictions.csv")
    ap.add_argument("--out_json", type=str, default="result/metrics.json")
    ap.add_argument("--no_intervals", action="store_true", help="Disable conformal intervals even if q_hat available.")
    ap.add_argument("--system_id_mode", type=str, default="filename_prefix",
                    choices=["filename_prefix", "parent_dir"],
                    help="Heuristic for system id extraction.")
    ap.add_argument("--filename_delim", type=str, default="_",
                    help="Delimiter when using filename_prefix mode.")
    ap.add_argument("--filename_prefix_index", type=int, default=0,
                    help="Index after splitting filename by delimiter.")
    return ap.parse_args()


def forward_feature(model, wave):
    with torch.no_grad():
        wav_embed = model.encode_clap_audio(wave)
        if wav_embed.dim() > 2:
            wav_embed = wav_embed.mean(dim=1)
    return wav_embed


def extract_system_id(path, mode="filename_prefix", delim="_", idx=0):
    fname = os.path.basename(path)
    stem = os.path.splitext(fname)[0]
    if mode == "parent_dir":
        return os.path.basename(os.path.dirname(path)) or "UNKNOWN"
    parts = stem.split(delim)
    if idx < len(parts):
        return parts[idx]
    return parts[0]


@torch.no_grad()
def predict(model, head, loader, device):
    """
    Run prediction over loader.
    Supports dataset batches shaped as:
      (waves, mos)  OR  (waves, mos, meta)
    Attempts to recover file paths from:
      1) meta (list or dict with 'path')
      2) loader.dataset.files (if present)
      3) falls back to 'UNKNOWN'
    """
    head.eval()
    preds = []
    targets = []
    filepaths = []
    sample_index = 0  # track position for fallback path recovery

    ds_files = None
    if hasattr(loader, "dataset"):
        # Common attribute name used in many custom datasets
        if hasattr(loader.dataset, "files"):
            ds_files = getattr(loader.dataset, "files")

    for batch in loader:
        # Unpack flexible batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                waves, mos, meta = batch
            elif len(batch) == 2:
                waves, mos = batch
                meta = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        else:
            raise ValueError("Batch is not a tuple/list; cannot unpack.")

        if isinstance(waves, list):  # safety
            waves = torch.stack(waves, dim=0)

        waves = waves.to(device)
        mos = mos.to(device)

        wav_embed = forward_feature(model, waves)
        _, _, pred_mos = head(wav_embed)

        preds.append(pred_mos.cpu())
        targets.append(mos.cpu())

        batch_size = waves.size(0)

        # Resolve file paths
        if meta is not None:
            if isinstance(meta, list):
                filepaths.extend(meta)
            elif isinstance(meta, dict) and "path" in meta:
                # meta['path'] could be list or single path
                if isinstance(meta["path"], list):
                    filepaths.extend(meta["path"])
                else:
                    filepaths.extend([meta["path"]] * batch_size)
            else:
                # meta present but unusable
                if ds_files is not None:
                    filepaths.extend(ds_files[sample_index: sample_index + batch_size])
                else:
                    filepaths.extend(["UNKNOWN"] * batch_size)
        else:
            if ds_files is not None:
                filepaths.extend(ds_files[sample_index: sample_index + batch_size])
            else:
                filepaths.extend(["UNKNOWN"] * batch_size)

        sample_index += batch_size

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return filepaths, preds, targets


def compute_and_print_metrics(filepaths, preds, targets, system_ids=None):
    truth_overall_array = targets.astype(float)
    pred_overall_array = preds.astype(float)

    print("==========UTTERANCE===========")
    print("======OVERALL QUALITY=======")
    mse_u = np.mean((truth_overall_array - pred_overall_array) ** 2)
    print('[UTTERANCE] Test error= %f' % mse_u)
    lcc_u_mat = np.corrcoef(truth_overall_array, pred_overall_array)
    lcc_u = lcc_u_mat[0][1]
    print('[UTTERANCE] Linear correlation coefficient= %f' % lcc_u)
    srcc_u_res = scipy.stats.spearmanr(truth_overall_array.T, pred_overall_array.T)
    srcc_u = srcc_u_res[0]
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % srcc_u)
    ktau_u_res = scipy.stats.kendalltau(truth_overall_array, pred_overall_array)
    ktau_u = ktau_u_res[0]
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % ktau_u)

    print("==========SYSTEM===========")
    if system_ids is None:
        system_ids = np.array([f"SYS_{i}" for i in range(len(filepaths))])
    system_ids = np.array(system_ids)

    sys_truth = []
    sys_pred = []
    for sid in np.unique(system_ids):
        sel = system_ids == sid
        sys_truth.append(truth_overall_array[sel].mean())
        sys_pred.append(pred_overall_array[sel].mean())
    sys_truth = np.array(sys_truth)
    sys_pred = np.array(sys_pred)

    print("======OVERALL QUALITY=======")
    mse_s = np.mean((sys_truth - sys_pred) ** 2)
    print('[SYSTEM] Test error= %f' % mse_s)
    lcc_s_mat = np.corrcoef(sys_truth, sys_pred)
    lcc_s = lcc_s_mat[0][1]
    print('[SYSTEM] Linear correlation coefficient= %f' % lcc_s)
    srcc_s_res = scipy.stats.spearmanr(sys_truth.T, sys_pred.T)
    srcc_s = srcc_s_res[0]
    print('[SYSTEM] Spearman rank correlation coefficient= %f' % srcc_s)
    ktau_s_res = scipy.stats.kendalltau(sys_truth, sys_pred)
    ktau_s = ktau_s_res[0]
    print('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % ktau_s)

    metrics = {
        "utterance": {
            "overall": {
                "MSE": float(mse_u),
                "LCC": float(lcc_u),
                "SRCC": float(srcc_u),
                "KTAU": float(ktau_u),
            }
        },
        "system": {
            "overall": {
                "MSE": float(mse_s),
                "LCC": float(lcc_s),
                "SRCC": float(srcc_s),
                "KTAU": float(ktau_s),
            }
        }
    }
    return metrics


def build_loader(files, scores, batch_size, num_workers, preproc):
    loader = AudioLoader()
    dataset = AudioMOSDataset(files, scores, loader, preproc)
    # Modify AudioMOSDataset __getitem__ to also return the original path if not already.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    test_files, test_scores = load_mos_txt(args.test_list, args.wav_dir)
    preproc = Preprocessor(T=16000 * 5)
    test_loader = build_loader(test_files, test_scores, args.batch_size, args.num_workers, preproc)
    print(f"Loaded {len(test_files)} test samples.")
    print(test_loader)

    # Upstream model (frozen)
    UPSTREAM_OUT_DIM = 768
    model = PortableM2D(weight_file=args.weight, flat_features=True).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Prediction head
    head = PredictionHead(in_dim=UPSTREAM_OUT_DIM, num_bins=20).to(device)

    q_hat = None
    alpha = None
    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "head_state" in ckpt:
            head.load_state_dict(ckpt["head_state"])
        q_hat = ckpt.get("q_hat", None)
        alpha = ckpt.get("alpha", None)
        print(f"Loaded checkpoint: {args.checkpoint} | q_hat={q_hat} | alpha={alpha}")
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found. Using randomly initialized head.")

    # Predict
    filepaths, preds, targets = predict(model, head, test_loader, device)

    # Conformal intervals
    if (q_hat is not None) and (not args.no_intervals):
        lower = np.clip(preds - q_hat, 1.0, 5.0)
        upper = np.clip(preds + q_hat, 1.0, 5.0)
    else:
        lower = np.full_like(preds, np.nan)
        upper = np.full_like(preds, np.nan)

    # System IDs
    system_ids = [
        extract_system_id(fp,
                          mode=args.system_id_mode,
                          delim=args.filename_delim,
                          idx=args.filename_prefix_index)
        for fp in filepaths
    ]

    # Metrics (textual alignment not available -> None)
    metrics = compute_and_print_metrics(
        filepaths=filepaths,
        preds=preds,
        targets=targets,
        system_ids=system_ids
    )

    # Save predictions CSV
    header = "filepath,system_id,truth_overall,pred_overall,lower,upper"
    with open(args.out_csv, "w") as f:
        f.write(header + "\n")
        for fp, sid, t, p, lo, up in zip(filepaths, system_ids, targets, preds, lower, upper):
            f.write(f"{fp},{sid},{t:.4f},{p:.4f},{lo:.4f},{up:.4f}\n")
    print(f"Saved predictions to {args.out_csv}")

    # Save metrics JSON
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()