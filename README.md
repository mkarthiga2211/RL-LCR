# RL-LCR: Reinforcement Learning with Lagrangian Clinical Constraints for Alarm-Aware Routing

This repository accompanies the manuscript "RL-LCR: Reinforcement Learning with Lagrangian Clinical Constraints for Alarm-Aware Routing in ICU and Ambulatory Sensor Networks". RL-LCR routes physiological waveform traffic over IEEE 802.15.4 networks. A Q-learning agent works with a primal-dual controller so that critical alarms stay within their delay budgets and wearable nodes stay within their energy limits.

## Repository status

This repository holds a partial set of scripts covering the main pipeline. The datasets and large files are too big to host on GitHub. The full code, configuration files and supporting data can be downloaded here:

**Full package:** https://drive.google.com/file/d/1OJs0MHmszvi38UQawqMFwV7Ksy3FPvUI/view?usp=sharing

## Contents

```
RL-LCR/
├── preprocessing/   # window extraction, quality rules, filtering
├── labelling/       # alarm labelling with persistence rules
├── traffic/         # packetisation and offered-load generation
├── routing/         # Q-learning agent and primal-dual controller
├── evaluation/      # metrics, multi-seed runs, statistical tests
└── README.md
```

## Datasets

The raw datasets are not redistributed. Download them from PhysioNet and the UCI repository under their own licences:

- MIMIC-III Waveform Database: https://physionet.org/content/mimic3wdb/ (credentialed access)
- eICU Collaborative Research Database: https://physionet.org/content/eicu-crd/ (credentialed access)
- BIDMC PPG and Respiration Dataset: https://physionet.org/content/bidmc/
- PPG-DaLiA: https://archive.ics.uci.edu/dataset/495/ppg+dalia

## Requirements

- Python 3.10 or later
- wfdb, numpy, scipy, pandas, matplotlib
- PyTorch (the primal solver was run with the MPS backend; CPU and CUDA also work)

```
pip install wfdb numpy scipy pandas matplotlib torch
```

## Usage

1. Download the full package from the Drive link above and extract it into the repository root.
2. Place the datasets in `data/` as described in the package.
3. Run preprocessing, then training, then evaluation using the scripts in each folder.

## Citation

If you use this code, please cite the manuscript (details will be added after publication).

## License

MIT License
