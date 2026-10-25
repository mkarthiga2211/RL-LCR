# RL-LCR: Reinforcement-Learning Lagrangian-Constrained Routing

Reinforcement-Learning Lagrangian-Constrained Routing for software-defined wireless sensor and body-area networks (SD-WSN/WBAN) used in ICU and ambulatory patient-vitals monitoring.

## Overview

RL-LCR couples a tabular Q-learning routing agent with a fixed-multiplier Lagrangian gradient-descent optimization layer for SD-WSN/WBAN patient-vitals monitoring:
- Proposes $K$ candidate routes ranked by tabular Q-values.
- Adjusts a relaxed route variable $x$ to satisfy strict end-to-end delay constraints $D(x) \le D_{\max}[\text{alarm class}]$ (with ARRHYTHMIA, DESATURATION, and HYPOTENSION tighter than ROUTINE).
- Enforces node energy draw limits $E \le E_{\max}[\text{node type}]$ (distinguishing bedside mains-assisted vs wearable battery-only nodes).
- Verifies feasibility before the SDN controller installs flow rules; if infeasible, iterates to the next-best RL candidate.

## Setup (Apple Silicon)

Optimized for macOS Apple Silicon (arm64, unified memory) using PyTorch Metal Performance Shaders (MPS):

```bash
# Create conda environment with Python 3.11
make env
conda activate rllcr

# Install package in editable mode
make install

# Configure environment variables
cp .env.example .env
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify environment
make check-env
```

## Data Access

The project uses the following datasets:
- **Primary Fitting**: MIMIC-III Waveform Matched Subset and MIMIC-III Clinical Database (PhysioNet credentialed access required).
- **Zero-Shot External Evaluation**: eICU, BIDMC, PPG-DaLiA, and WESAD (optional).

Set credentials in `.env`:
```bash
PHYSIONET_USERNAME=your_username
DATA_ROOT=data
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Pipeline

The end-to-end experimental pipeline progresses through:
1. `00_check_env`: Validate environment, MPS fallback, and hardware P-cores.
2. Data collection and cataloging (`mimic_download.py`, `mimic_catalog.py`).
3. Preprocessing and signal cleaning (`cleaning.py`, `events.py`, `splits.py`).
4. Network simulation and trace generation (`sim/`, `trace_player.py`).
5. RL-LCR and baseline protocol evaluation (`routing/`, `eval/`).
6. Result verification, table generation, and manuscript figure rendering (`viz/`, `reports/`).

## Reproduce

Run automated linting, test suites, and benchmarks:

```bash
# Run code quality checks
make lint

# Run fast unit tests
make test

# Run full test suite
make test-all
```

## Citation

```bibtex
@article{rllcr2026,
  title={RL-LCR: Reinforcement-Learning Lagrangian-Constrained Routing for SD-WSN/WBAN Patient-Vitals Monitoring},
  author={Research Team},
  year={2026}
}
```
