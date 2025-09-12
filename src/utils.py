import os
import warnings
import torch, torchaudio
import datetime
from torch.utils.data.dataset import Dataset

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