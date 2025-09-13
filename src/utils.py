import os
import warnings
import torch, torchaudio
import datetime
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)


def load_mos_txt(txt_path, wav_dir):
    """
    Load audio file paths and MOS values from a txt file.
    Only include files that exist in wav_dir.
    txt file format: <audio_filename>,<mos>
    Returns: list of full audio paths, list of MOS scores
    """
    audio_files = []
    mos_scores = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                fname, mos = parts
                full_path = os.path.join(wav_dir, fname)
                if os.path.isfile(full_path):
                    audio_files.append(full_path)
                    mos_scores.append(float(mos))
                # else:
                    # print(f"Warning: File not found and skipped: {full_path}")
    return audio_files, mos_scores



# 1. Input Audio File
class AudioLoader:
    '''
    Load audio file and convert to mono waveform tensor.
    '''
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    def load(self, filepath):
        waveform, sr = torchaudio.load(filepath, backend="soundfile")
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform.mean(dim=0)  # mono
    

# 2. Preprocessing
class Preprocessor:
    '''
    Process waveform to fixed length T (in samples).
    If shorter, pad with zeros. If longer, truncate.
    '''
    def __init__(self, T):
        self.T = T
    def process(self, waveform):
        length = waveform.shape[-1]
        if length > self.T:
            return waveform[:self.T]
        elif length < self.T:
            pad = torch.zeros(self.T - length, dtype=waveform.dtype)
            return torch.cat([waveform, pad])
        return waveform
    
# 3. Dataset Class
class AudioMOSDataset(Dataset):
    def __init__(self, audio_files, mos_scores, loader, preproc):
        self.audio_files = audio_files
        self.mos_scores = mos_scores
        self.loader = loader
        self.preproc = preproc

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform = self.loader.load(self.audio_files[idx])  #torch.Size([36190]) torch.Size([38075])
        proc_wave = self.preproc.process(waveform)    #torch.Size([80000])
        mos = torch.tensor(self.mos_scores[idx], dtype=torch.float32)
        return proc_wave, mos




def create_logger(log_path: str):
    """Create logger. If log_path is a directory or has no extension, place a timestamped file inside."""
    if not log_path:
        return print, None
    # Determine if user supplied a directory (existing or no file extension)
    if os.path.isdir(log_path) or os.path.splitext(os.path.basename(log_path))[1] == "":
        os.makedirs(log_path, exist_ok=True)
        log_path = os.path.join(
            log_path,
            datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S.txt")
        )
    else:
        # Ensure parent directory exists (use '.' if no parent)
        parent = os.path.dirname(log_path) or "."
        os.makedirs(parent, exist_ok=True)
    f = open(log_path, "a", buffering=1)
    abs_path = os.path.abspath(log_path)
    print(f"[logger] Logging to {abs_path}")
    def _log(*args, **kwargs):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        f.write(f"[{ts}] {msg}\n")
        f.flush()
        print(*args, **kwargs)
    return _log, f


def forward_feature(model, wave, device, keep_sequence=True):
    with torch.no_grad():
        # wave = torch.Size([32, 1, 80000])
        if hasattr(model, "encode_clap_audio"): # M2D interface
            feats = model.encode_clap_audio(wave)
        else: # Wav2Vec2 interface
            # Assume HuggingFace Wav2Vec2Model style: input (B,T)
            if wave.dim() == 3 and wave.size(1) == 1:
                wave = wave.squeeze(1) 
            wave = wave.to(device).float()
            out = model(wave)
            feats = out.last_hidden_state    # (B, T', D)
        # if not keep_sequence and feats.dim() > 2:
        #     feats = feats.mean(dim=1)
    return feats


def export_learning_curves(metrics, out_dir, upstream, last_saved_epoch, log):
    """
    Save metrics.csv and learning_curve_<upstream>.png. Returns dict with paths and best_epoch.
    """
    if not metrics.get("train_mse"):
        return None
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "valid_mse", "q_hat"])
        for i in range(len(metrics["train_mse"])):
            w.writerow([
                i + 1,
                metrics["train_mse"][i],
                metrics["valid_mse"][i] if i < len(metrics["valid_mse"]) else "",
                metrics["q_hat"][i] if i < len(metrics["q_hat"]) else ""
            ])

    # Best/early-stop epoch
    if metrics.get("valid_mse"):
        if last_saved_epoch is not None:
            best_epoch = last_saved_epoch
        else:
            best_epoch = min(range(len(metrics["valid_mse"])), key=lambda i: metrics["valid_mse"][i]) + 1
    else:
        best_epoch = len(metrics["train_mse"])

    # Plot
    epochs = list(range(1, len(metrics["train_mse"]) + 1))
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, metrics["train_mse"], label="Train MSE")
    if metrics.get("valid_mse"):
        plt.plot(epochs, metrics["valid_mse"], label="Valid MSE")
    plt.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best/Early stop @ {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Learning Curves ({upstream})")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(out_dir, f"learning_curve_{upstream}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    log(f"[PLOT] Saved learning curve to {png_path}")
    log(f"[PLOT] Saved metrics CSV to {csv_path}")
    return {"csv_path": csv_path, "png_path": png_path, "best_epoch": best_epoch}


def apply_intervals(preds: torch.Tensor, q_hat: float, low=1.0, high=5.0):
    lower = (preds - q_hat).clamp(min=low, max=high)
    upper = (preds + q_hat).clamp(min=low, max=high)
    return lower, upper