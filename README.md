# Anonymous Reproducibility Package

This repository provides an anonymized reproducibility package for a double-blind submission on deployment-time routing for vision pipelines.

## Contents

### `care/`
Reference implementation of the routing module used in the submission.

### `bo_calibration/`
Scenario-conditional calibration code for utility-weight tuning.

### `eval_results/`
JSON outputs for the experiments reported in the submission, including:
- cross-hardware latency probe
- non-JPEG degradation validation
- slicing head-to-head comparison

## Reproducibility Scope

This package contains the routing implementation, calibration code, and result files needed to reproduce the reported decision-support and latency analyses.

## License

MIT
