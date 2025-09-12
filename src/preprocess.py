
import torch, torchaudio
from typing import List
from torch.utils.data import DataLoader, Sampler, Dataset
from utils import load_mos_txt, AudioLoader, Preprocessor, AudioMOSDataset, create_logger

# --- Simple Augmentation Utilities ---
class SimpleAugment:
    """
    Lightweight, MOS-preserving augmentations:
      - Small random gain (-1dB .. +1dB)
      - Mild noise (SNR 15..30 dB)
      - Time shift (<=5% length)
      - Time stretch (0.95x .. 1.05x) if torchaudio available
    Each applied with independent probability p.
    """
    def __init__(self, p=0.5, max_gain_db=1.0, noise_snr_range=(15, 30),
                 max_shift_ratio=0.05, stretch_factor_range=(0.95, 1.05)):
        self.p = p
        self.max_gain_db = max_gain_db
        self.noise_snr_range = noise_snr_range
        self.max_shift_ratio = max_shift_ratio
        self.stretch_factor_range = stretch_factor_range

    def _maybe(self):
        return torch.rand(1).item() < self.p

    def _ensure_shape(self, wav):
        # Accept (T) or (1,T); convert to (1,T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return wav

    def gain(self, wav):
        if not self._maybe(): return wav
        db = torch.empty(1).uniform_(-self.max_gain_db, self.max_gain_db).item()
        factor = 10 ** (db / 20)
        return (wav * factor).clamp(-1.0, 1.0)

    def noise(self, wav):
        if not self._maybe(): return wav
        rms = wav.pow(2).mean().sqrt()
        if rms < 1e-6:
            return wav
        snr_db = torch.empty(1).uniform_(*self.noise_snr_range).item()
        noise_rms = rms / (10 ** (snr_db / 20))
        noise = torch.randn_like(wav) * noise_rms
        return (wav + noise).clamp(-1.0, 1.0)

    def shift(self, wav):
        if not self._maybe(): return wav
        T = wav.size(-1)
        max_shift = int(T * self.max_shift_ratio)
        if max_shift < 1:
            return wav
        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if shift == 0:
            return wav
        return torch.roll(wav, shifts=shift, dims=-1)

    def stretch(self, wav, sr=16000):
        if torchaudio is None or not self._maybe():
            return wav
        factor = torch.empty(1).uniform_(*self.stretch_factor_range).item()
        if abs(factor - 1.0) < 1e-3:
            return wav
        # Resample approach (simple & fast). Keeps pitch slightly altered but minor for small factors.
        orig_len = wav.size(-1)
        new_len = int(orig_len / factor)
        if new_len <= 16:
            return wav
        wav_stretch = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=int(sr * factor))
        # Center-crop or pad to original length
        if wav_stretch.size(-1) > orig_len:
            start = (wav_stretch.size(-1) - orig_len) // 2
            wav_stretch = wav_stretch[..., start:start + orig_len]
        elif wav_stretch.size(-1) < orig_len:
            pad = orig_len - wav_stretch.size(-1)
            wav_stretch = torch.nn.functional.pad(wav_stretch, (0, pad))
        return wav_stretch

    def __call__(self, wav):
        wav = self._ensure_shape(wav)
        wav = self.gain(wav)
        wav = self.noise(wav)
        wav = self.shift(wav)
        wav = self.stretch(wav)
        return wav

class AugmentedDataset(Dataset):
    """Wrap an AudioMOSDataset to apply augmentation on-the-fly (only for training)."""
    def __init__(self, base_dataset, augment_fn):
        self.base = base_dataset
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        wav, mos = self.base[idx]
        try:
            wav = self.augment_fn(wav)
        except Exception:
            # Fail-safe: return original waveform
            pass
        return wav, mos
# --- End Augmentation Utilities ---


# -------------- Dataset Functions --------------
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
                  batch_size_train=32, batch_size_eval=8, num_workers=4,
                  stratified=False, strat_bins=5, augment=None):
    trainset = AudioMOSDataset(train_files, train_scores, loader, preproc)
    if augment is not None:
        trainset = AugmentedDataset(trainset, augment)
    calibset = AudioMOSDataset(calib_files, calib_scores, loader, preproc)
    validset = AudioMOSDataset(valid_files, valid_scores, loader, preproc)

    if stratified:
        print("Using stratified batching...")
        sampler = StratifiedBatchSampler(train_scores, batch_size=batch_size_train, n_bins=strat_bins, quantile=True)
        trainloader = DataLoader(trainset, batch_sampler=sampler, num_workers=num_workers)
    else:
        trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)

    calibloader = DataLoader(calibset, batch_size=batch_size_eval, shuffle=False, num_workers=max(1, num_workers//2))
    validloader = DataLoader(validset, batch_size=batch_size_eval, shuffle=False, num_workers=max(1, num_workers//2))
    return trainloader, calibloader, validloader


class StratifiedBatchSampler(Sampler):
    """
    Build balanced mini-batches across MOS strata (quantile bins).
    """
    def __init__(self, mos_values, batch_size, n_bins=5, drop_last=False, quantile=True, seed=0):
        self.mos_values = torch.as_tensor(mos_values, dtype=torch.float)
        self.batch_size = batch_size
        self.n_bins = n_bins
        self.drop_last = drop_last
        self.quantile = quantile
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        if quantile:
            qs = torch.linspace(0, 1, n_bins + 1)
            self.bin_edges = torch.quantile(self.mos_values, qs)
            # ensure strictly increasing (handle duplicates)
            for i in range(1, len(self.bin_edges)):
                if self.bin_edges[i] <= self.bin_edges[i-1]:
                    self.bin_edges[i] = self.bin_edges[i-1] + 1e-4
        else:
            lo, hi = float(self.mos_values.min()), float(self.mos_values.max()) + 1e-6
            self.bin_edges = torch.linspace(lo, hi, n_bins + 1)

        self.bin_indices = []
        for b in range(n_bins):
            left, right = self.bin_edges[b], self.bin_edges[b+1]
            if b < n_bins - 1:
                mask = (self.mos_values >= left) & (self.mos_values < right)
            else:
                mask = (self.mos_values >= left) & (self.mos_values <= right)
            idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            self.bin_indices.append(idxs)

    def _generate_batches(self):
        # shuffle each bin
        shuffled = []
        for idxs in self.bin_indices:
            t = torch.tensor(idxs)
            if len(t) > 0:
                perm = t[torch.randperm(len(t), generator=self.rng)].tolist()
            else:
                perm = []
            shuffled.append(perm)

        ptrs = [0] * self.n_bins
        batches = []
        current = []
        bin_cycle = list(range(self.n_bins))

        while True:
            progress = False
            for b in bin_cycle:
                if ptrs[b] < len(shuffled[b]):
                    current.append(shuffled[b][ptrs[b]])
                    ptrs[b] += 1
                    progress = True
                    if len(current) == self.batch_size:
                        batches.append(current)
                        current = []
                if all(ptrs[i] >= len(shuffled[i]) for i in range(self.n_bins)):
                    break
            if not progress:
                break

        if len(current) and not self.drop_last:
            batches.append(current)
        return batches

    def __iter__(self):
        for batch in self._generate_batches():
            yield batch

    def __len__(self):
        total = len(self.mos_values)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size