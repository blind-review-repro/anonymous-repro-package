# Anonymous Reproducibility Package

This package supports the manuscript "Resolution Confounding in SR-Assisted UAV Detection: A Controlled Empirical Study and a Cost-Aware Routing Expert".

## Contents

### `care/care_router.py`
Reference implementation of CARE (Algorithm 1 in the paper). Implements the two-stage inference engine (feasibility filtering + utility maximization) with explanation trace generation.

Run: `python care/care_router.py`

### `bo_calibration/bo_calibration.py`
Scenario-conditional utility calibration via Bayesian Optimization (Section in the paper). Reproduces Table (BO calibration results) and Figures (convergence + scenario comparison).

Run: `python bo_calibration/bo_calibration.py`

Outputs:
- `eval_results/bo_calibration.json`
- `figures/bo_convergence.pdf`
- `figures/bo_scenario_comparison.pdf`

### `latency_profiling/`
Scripts for measuring per-image latency on target hardware. See `profile_laptop.py` for the cross-hardware probe used in the RTX 5060 validation.

### `eval_results/`
JSON outputs from the experiments reported in the paper:
- `laptop_latency_nvidia_geforce_rtx_5060_laptop_gpu.json` — 5060 cross-hardware latency probe
- `nonjpeg_*.json` — non-JPEG degradation validation (motion blur, Gaussian noise)
- `sahi_headtohead.json` — SAHI slicing head-to-head comparison

## Key Tables in the Paper (and how to reproduce)

| Paper Table | Source |
|---|---|
| Table (cross-hardware latency) | `eval_results/laptop_latency_*.json` |
| Table (non-JPEG degradation) | `eval_results/nonjpeg_*.json` |
| Table (SAHI head-to-head) | `eval_results/sahi_headtohead.json` |
| Table (BO calibration) | Run `bo_calibration.py` → `eval_results/bo_calibration.json` |

## Hardware Notes

- Main experiments: 4x RTX 4090 (48GB each)
- Cross-hardware probe: RTX 5060 Laptop GPU (8.55GB, Blackwell)
- All latency values are per-image, warmup=10, runs=30

## License

MIT
