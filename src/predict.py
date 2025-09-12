import sys
import os
import argparse
import json
import numpy as np
import torch
import scipy.stats
import datetime
from torch.utils.data import DataLoader

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_mos_txt, AudioLoader, Preprocessor, AudioMOSDataset
from NNmodel import PredictionHead, TransformerPredictionHead


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
    ap.add_argument("--out_result", type=str, default="result")
    ap.add_argument("--no_intervals", action="store_true", help="Disable conformal intervals even if q_hat available.")
    ap.add_argument("--system_id_mode", type=str, default="filename_prefix",
                    choices=["filename_prefix", "parent_dir"],
                    help="Heuristic for system id extraction.")
    ap.add_argument("--filename_delim", type=str, default="_",
                    help="Delimiter when using filename_prefix mode.")
    ap.add_argument("--filename_prefix_index", type=int, default=0,
                    help="Index after splitting filename by delimiter.")
    ap.add_argument("--cond_bins", type=int, default=4,
                    help="Quantile bins for conditional coverage (0 disables).")
    ap.add_argument("--use_transformer_head", action="store_true",
                    help="Use single-layer transformer + attention pooling head.")
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
    # Ensure predict() can recover file paths
    if not hasattr(dataset, "files"):
        dataset.files = files
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def main():
    args = parse_args()
    os.makedirs(args.out_result, exist_ok=True) 
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
    head_cls = TransformerPredictionHead if args.use_transformer_head else PredictionHead
    print('Using head class: ' + str(head_cls))
    head = head_cls(in_dim=UPSTREAM_OUT_DIM, num_bins=20).to(device)

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
    # print(filepaths[:5], preds[:5], targets[:5])

    # Conformal intervals
    if (q_hat is not None) and (not args.no_intervals):
        print(f"Using conformal intervals with q_hat={q_hat:.4f}")
        lower = np.clip(preds - q_hat, 1.0, 5.0)
        upper = np.clip(preds + q_hat, 1.0, 5.0)
    else:
        lower = np.full_like(preds, np.nan)
        upper = np.full_like(preds, np.nan)

    # Interval stats (utterance-level)
    interval_stats = {}
    if not np.isnan(lower).all():
        mask = ~np.isnan(lower)
        coverage = float(np.mean((targets[mask] >= lower[mask]) & (targets[mask] <= upper[mask])))
        widths = upper[mask] - lower[mask]
        avg_width = float(np.mean(widths))
        nominal = float(1 - alpha) if alpha is not None else None
        calibration_error = float(abs(coverage - nominal)) if nominal is not None else None

        radii = widths / 2.0
        sharpness_msr = float(np.mean(radii ** 2))
        sharpness_rmsr = float(np.sqrt(sharpness_msr))
        sharpness_norm = float(avg_width / 4.0)

        interval_stats = {
            "coverage": coverage,
            "avg_width": avg_width,
            "nominal": nominal,
            "calibration_error": calibration_error,
            "sharpness": {
                "mean_squared_radius": sharpness_msr,
                "rms_radius": sharpness_rmsr,
                "normalized_width": sharpness_norm
            }
        }

        # Conditional coverage (by true MOS & predicted MOS quantiles)
        if args.cond_bins > 1:
            def quantile_bins(arr, k):
                qs = np.linspace(0, 1, k + 1)
                edges = np.unique(np.quantile(arr, qs))
                # Guarantee at least 2 edges
                if len(edges) < 2:
                    return [float(arr.min()), float(arr.max())]
                return edges

            def bin_coverage(val_source, name):
                edges = quantile_bins(val_source, args.cond_bins)
                bin_cov = []
                bin_cnt = []
                bins_list = []
                for i in range(len(edges) - 1):
                    lo_e, hi_e = edges[i], edges[i + 1]
                    sel = (val_source >= lo_e) & (val_source <= (hi_e if i == len(edges) - 2 else hi_e))
                    cnt = int(sel.sum())
                    if cnt == 0:
                        cov = None
                    else:
                        cov = float(np.mean((targets[sel] >= lower[sel]) & (targets[sel] <= upper[sel])))
                    bin_cov.append(cov)
                    bin_cnt.append(cnt)
                    bins_list.append([float(lo_e), float(hi_e)])
                gaps = [abs(c - coverage) for c in bin_cov if c is not None]
                max_gap = float(max(gaps)) if gaps else None
                return {
                    "bins": bins_list,
                    "coverage": bin_cov,
                    "counts": bin_cnt,
                    "max_abs_gap": max_gap
                }

            interval_stats["conditional"] = {
                "true_mos": bin_coverage(targets, "true"),
                "pred_mos": bin_coverage(preds, "pred")
            }
        # (Optional) length-based coverage could be added if variable lengths available.

        print(
            f"[INTERVAL] coverage={coverage:.4f} "
            f"nominal={(nominal if nominal is not None else 'NA')} "
            f"calib_error={(calibration_error if calibration_error is not None else 'NA')} "
            f"avg_width={avg_width:.4f} "
            f"sharp_msr={sharpness_msr:.4f} sharp_rmsr={sharpness_rmsr:.4f} "
            f"sharp_norm={sharpness_norm:.4f}"
        )
        if "conditional" in interval_stats:
            tc = interval_stats["conditional"]["true_mos"]
            pc = interval_stats["conditional"]["pred_mos"]
            print(f"[INTERVAL][COND] true_mos max_gap={tc['max_abs_gap']} pred_mos max_gap={pc['max_abs_gap']}")
    else:
        print("[INTERVAL] No intervals computed (q_hat missing or disabled).")

    # System IDs
    system_ids = [
        extract_system_id(fp,
                          mode=args.system_id_mode,
                          delim=args.filename_delim,
                          idx=args.filename_prefix_index)
        for fp in filepaths
    ]

    # Metrics
    metrics = compute_and_print_metrics(
        filepaths=filepaths,
        preds=preds,
        targets=targets,
        system_ids=system_ids
    )

    if interval_stats:
        metrics["utterance"]["overall"]["intervals"] = interval_stats

    # Save predictions CSV
    timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    header = "filepath,system_id,truth_overall,pred_overall,lower,upper"
    out_csv = os.path.join(args.out_result, f"predictions_{timestamp}.csv")
    with open(out_csv, "w") as f:
        f.write(header + "\n")
        for fp, sid, t, p, lo, up in zip(filepaths, system_ids, targets, preds, lower, upper):
            f.write(f"{fp},{sid},{t:.4f},{p:.4f},{lo:.4f},{up:.4f}\n")
    print(f"Saved predictions to {out_csv}")

    # Save metrics JSON
    out_json = os.path.join(args.out_result, f"metrics_{timestamp}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_json}")


if __name__ == "__main__":
    main()

# python src/predict.py --use_transformer_head