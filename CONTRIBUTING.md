# Contributing

Use Python 3.12 and the public dependencies in `requirements-dev.txt`. Run `python -m pytest tests -q` before proposing changes. Public inference lives in `conformalmos/`; historical experiments live in `src/` and should be treated as archival references.

For a bug report include the failing command, operating system, Python and PyTorch versions, and complete traceback. For prediction discrepancies include the input sample rate/duration, checkpoint hashes and calibration metadata. Share audio only when you have permission.

Please keep calibration, model-selection and evaluation data separate. Do not replace historical results with newly corrected quantile results without identifying the methodological change. Dataset access, full-paper reproducibility, historical checkpoint provenance and current licensing limitations are documented in README and the model card.
