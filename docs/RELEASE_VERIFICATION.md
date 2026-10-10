# Public release verification — 2026-10-10

## Executed checks

- Created an isolated Python 3.12 virtual environment under `/tmp`; installed the CPU PyTorch 2.8.0 / torchaudio 2.8.0 / torchvision 0.23.0 wheels and declared release dependencies. No research-environment packages were changed.
- `python -m pytest tests -q`: **15 passed** after correcting the test checkpoint loader to use CPU mapping. Repeated from a fresh local Git clone of the release commit: **15 passed**; CLI help also passed there.
- `python -m pip check`: **No broken requirements found**.
- `python -m conformalmos.predict --help`: passed.
- Pinned official M2D runtime download with `python scripts/download_m2d.py --runtime-only`: completed; SHA-256 matched the manifest.
- Full encoder/head CPU inference on one accessible local BVCC recording: completed. MOS = **3.775054931641**; historical-radius bounds = [2.967670500278473, 4.582439363002777]; nominal setting = 90%.
- The public and historical local runtimes produced the same prediction on that recording: absolute difference **0.0**. This is a single-recording compatibility check, not a full numerical equivalence proof.
- A public-files-only export under `/tmp` ran inference in the isolated CPU environment, without `src/`, UTMOS, the original Python environment, or original head-directory imports. Prediction agreed within 1e-5.
- Tests cover finite-sample ranks at calibration sizes 25, 165 and 225; insufficient-sample infinite radius; historical interpolation; invalid residuals; clipping; real head loading; stereo conversion/resampling/padding/truncation; and calibration/evaluation path overlap rejection.

## Scope and remaining limitations

The large M2D checkpoint was read from the existing local installation; its bytes were verified against the manifest. The full upstream weight archive was **not downloaded again**. Therefore end-to-end archive download/extraction has not been exercised locally, although the runtime download and existing-weight hash checks have been exercised. The download helper does not accept weights until their checksum matches.

GitHub Actions has been configured but has not yet run on GitHub. CUDA inference was not independently validated in a clean CUDA environment. No full-dataset or paper-table reproduction was run during this release task. No new scientific performance claim is supported by the smoke tests.

The bundled head retains its historical interpolated calibration radius. New finite-sample calibration is a separate option. Path checks do not detect shared audio under different filenames or group/content leakage.

Following the author's explicit approval, the ConformalMOS code, release documentation, and bundled MOS head are licensed under MIT. Third-party materials retain their own terms; see `LICENSE_STATUS.md`.

## Preservation

Historical `src/` files, `predict.sh`, notebooks, datasets, and source checkpoints were not edited by this release task. Existing user modifications were left in place. Replaced README, requirements and ignore-file versions were backed up outside this repository in the sibling `ConvM2D2V2_release_backup_20261010/` folder. The bundled head is a byte-for-byte copy, not a modified checkpoint. ICASSP follow-up artifacts were not changed.
