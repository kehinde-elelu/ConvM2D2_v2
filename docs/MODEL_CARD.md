# ConformalMOS baseline model card

## Provenance

The bundled head is an unchanged copy of the September 17, 2025 baseline checkpoint identified in `assets/manifest.json`. Source architecture: `src/NNmodel.py`, class `PredictionHead`. Public implementation preserves parameter names and calculations but removes optional experimental dependencies.

Encoder: frozen M2D-CLAP `m2d_clap_vit_base-80x608p16x16-240128`. Encoder outputs are temporally averaged when three-dimensional. Head: LayerNorm(768), Linear(768,256), ReLU, Dropout(0.3), Linear(256,20). MOS is the expected value over 20 equally spaced bins from 1 to 5. The historical training uses Gaussian-smoothed ordinal labels; public inference does not retrain the model.

Hashes, upstream download URLs and historical interval alpha are recorded in `assets/manifest.json`. The upstream runtime is downloaded rather than relicensed or bundled. The runtime revision differs from the locally modified research runtime; release verification checks their actual prediction agreement on a local recording.

## Intended use and limitations

Research on synthetic-speech MOS prediction. A MOS estimate is not a human rating, a safety decision, or an objective assessment of every perceptual dimension. Only the opening five seconds are used. Language, speaker, recording conditions, synthesizer and collection shifts can change errors and interval coverage.

Intervals are utterance-level and clipped to [1,5]. System-mean accuracy and interval coverage are different metrics. Historical calibration uses interpolation. New finite-sample recalibration is a separately documented procedure requiring held-out labels and exchangeability. Neither provides arbitrary-domain or group-conditional guarantees.

The accessible local reproduction subset differs from the complete BVCC population reported in the paper; exact full-paper reproducibility is not asserted. Full dataset manifests and audio are not distributed in this release. No follow-up improvement is claimed.

## Data and upstream terms

Obtain BVCC through its official distribution and follow its data terms. Do not infer permission to redistribute audio from the presence of this code. M2D and optional historical UTMOS components retain their upstream licenses. The bundled head is an author-supplied research artifact.
