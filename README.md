# ConformalMOS

Public inference release for **ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling**, Interspeech 2026.

**[Final paper](https://www.isca-archive.org/interspeech_2026/elelu26_interspeech.pdf)** · [Citation](CITATION.cff) · [Model card](docs/MODEL_CARD.md)

ConformalMOS predicts human Mean Opinion Scores (MOS) on the 1–5 scale using a frozen M2D-CLAP encoder and an ordinal prediction head. This release includes the small trained baseline head, unlabeled-audio inference, and explicit recalibration. It keeps the historical research scripts under `src/` separate from the supported public entry points.

## Quick start

Python **3.12** is tested. Linux CPU inference is supported; CUDA is optional. Run commands from the repository root.

```bash
git clone https://github.com/kehinde-elelu/ConvM2D2_v2.git
cd ConvM2D2_v2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU wheels; for CUDA, install matching torch/torchaudio/torchvision wheels instead.
python -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/download_m2d.py
python -m conformalmos.predict --input /path/to/audio.wav --output results/example.csv
```

The download helper retrieves the official M2D runtime at a pinned revision and the original M2D-CLAP archive (approximately GB scale). It checks SHA-256 hashes. Downloads go into ignored `external/`; the trained MOS head is included in `assets/`. Upstream M2D code and weights retain their own terms: [M2D license](https://github.com/nttcslab/m2d/blob/master/LICENSE.pdf).

For a directory of WAV files:

```bash
python -m conformalmos.predict --input /path/to/wavs --output results/predictions.csv --batch-size 8
# Optional GPU:
python -m conformalmos.predict --input /path/to/wavs --output results/gpu.csv --device cuda
```

Outputs contain `path,predicted_mos,lower,upper,nominal_coverage`; a companion `.csv.json` records model hashes and calibration provenance. Existing output files are protected. Use `--no-intervals` for point predictions only. No human MOS labels are needed for prediction.

Audio is averaged to mono, resampled to 16 kHz, truncated to its first five seconds or zero-padded to five seconds. Input amplitude is not normalized. Long recordings are therefore assessed using only their opening segment. Default inference runs on CPU with four threads.

## Prediction intervals and recalibration

The included checkpoint stores a **historical 90% nominal** calibration radius of `0.8073844313621521`, computed using an interpolated residual quantile. It is retained to preserve historical behavior. It is not the finite-sample-corrected split-conformal quantile and does not imply guaranteed coverage on arbitrary user audio.

To recalibrate, first predict a separate calibration collection, then join its human MOS labels into a CSV with columns `path,predicted_mos,mos`. List evaluation audio paths separately, one absolute path per line. Do not fit or select the model using calibration/evaluation labels.

```bash
python -m conformalmos.calibrate --csv calibration_predictions_with_labels.csv \
  --checkpoint assets/conformalmos_m2d_head.pt --evaluation-paths evaluation_paths.txt \
  --alpha 0.05 --output results/calibration95.json
python -m conformalmos.predict --input /path/to/evaluation_wavs \
  --calibration results/calibration95.json --output results/evaluation95.csv
```

Default recalibration uses the `ceil((n+1)*(1-alpha))`-th ordered absolute residual. If that rank exceeds the calibration size, the radius is infinite and clipped intervals cover the full MOS range. `--method historical` is available solely for historical comparisons.

The calibration command checks duplicate paths and calibration/evaluation path overlap. You must separately audit exact audio duplicates, training overlap, speakers, prompts, systems, and the prediction configuration. Marginal split-conformal validity requires exchangeability of calibration and future evaluation scores; it does not establish per-system coverage or transfer guarantees under distribution shift.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

CI runs CPU tests without downloading large M2D weights. Tests exercise the real head checkpoint, preprocessing, clipping, and corrected quantile edge cases. Full encoder inference verification and its limitations are recorded in [release verification](docs/RELEASE_VERIFICATION.md).

## Historical experiments and reproduction limits

`src/`, `predict.sh`, and notebooks are retained as historical research artifacts. They contain experimental variants, lab-specific paths, and optional UTMOS/wav2vec dependencies; they are not the public quick-start interface. `requirements-research.txt` lists additional packages, but does not claim to reproduce the full historical environment.

The included baseline head is copied byte-for-byte from `checkpoint_head_m2d_run_20250917_003744.pt`. It has been used in the local baseline reproduction; we do not certify that it is the checkpoint for every table in the paper. The paper reports BVCC results on a larger population than the currently accessible local subset. Its reported MSE of approximately 0.08 is a **system-level** result, not a promise of utterance-level performance on user audio. Full-paper reproduction requires the appropriate licensed BVCC audio, labels, exact original manifests, and experiment configuration; these are not bundled here.

Original audio datasets, other checkpoints, private environments, logs, and follow-up ICASSP experiments are not distributed. See [model card](docs/MODEL_CARD.md) for provenance and limitations.

## Citation

```bibtex
@inproceedings{elelu2026conformalmos,
  author = {Elelu, Kehinde and Siegel, Joshua E. and Saffary, Mohammadali and Ahmed, Tashfain and Babatunde, Simeon and Okpala, Ebuka},
  title = {{ConformalMOS}: Uncertainty-Aware {MOS} Prediction with Conformal Intervals and Ordinal Modeling},
  booktitle = {Interspeech 2026},
  year = {2026},
  url = {https://www.isca-archive.org/interspeech_2026/elelu26_interspeech.pdf}
}
```

For questions or reproducible bug reports, open a GitHub issue with the command, package versions, and error message. Please use audio you are permitted to share.

## License status

No software license has been selected yet. See [license status](docs/LICENSE_STATUS.md). Public testing instructions do not grant redistribution rights. Upstream M2D and UTMOS terms remain separate.
