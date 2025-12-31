# Experiments & Benchmarks

This directory contains the configurations, pipelines, and scripts to reproduce the results presented in the NeurIPS 2025 paper.

## 1. Ordered MNIST

A simple video prediction benchmark involving rotational dynamics.

*   **Dynamics:** Discrete-time, Normal.
*   **Data:** Automatically downloaded and generated via `torchvision`.
*   **Pipeline:** `experiments/pipelines/mnist.py`

**Running the experiment:**
```bash
./scripts/run_orderedmnist.sh
```

---

## 2. Chignolin Molecular Dynamics

We evaluate our method on the folding dynamics of the Chignolin mini-protein.

### Dataset Preparation
We use the dataset provided by [Marshall et al. (2024)](https://zenodo.org/records/10681926). Specifically, we use the trajectories generated with the **CHARMM22\*** force field (`C22star`), which corresponds to the setup described in our paper.

**Steps:**
1.  Download `trajs_out.zip` from [Zenodo](https://zenodo.org/records/10681926).
2.  Unzip the archive.
3.  Locate the `C22star` directory.
4.  Move or Symlink the contents of `C22star` to your data directory (e.g., `./data/chignolin_raw`).

**Expected Directory Structure:**
```text
data/
└── chignolin_raw/        # Matches 'C22star' content
    ├── protein.gro       # Topology
    ├── C0.xtc            # Trajectory 0
    ├── ...
    └── C33.xtc           # Trajectory 33
```

**Command Line Example:**

```bash
# Assuming you downloaded and unzipped trajs_out.zip
mkdir -p data/chignolin_raw

# Copy or Symlink the CHARMM22* data
cp -r /path/to/trajs_out/C22star/* data/chignolin_raw/

# Verify
ls data/chignolin_raw
# Should show: C0.xtc ... C33.xtc protein.gro ...

# Running the experiment:
./scripts/run_chignolin.sh
```

### Citation for Data
```bibtex
@dataset{marshall_2024_10681926,
  author       = {Marshall, Tim and Raddi, Robert and Voelz, Vincent},
  title        = {MD simulation trajectory data for "An Evaluation of Force Field Accuracy for the Mini-Protein Chignolin using Markov State Models"},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.10681926},
  url          = {https://doi.org/10.5281/zenodo.10681926},
}
```

---

## 3. Synthetic Benchmarks

We provide scripts to reproduce the synthetic experiments from the paper.

### Noisy Logistic Map
A discrete-time chaotic system with additive noise.
*   **Metric:** Directed Hausdorff Distance on the complex plane.
*   **Run:** `./scripts/run_logistic.sh`
*   **Config:** `experiments/configs/synthetic_logistic.py`

---

## 4. Upcoming Benchmarks

The following experiments are currently being refactored for the public release:
*   **ERA5 (Weather/Climate):** Application of Koopman operator to global weather data (ECMWF).
*   **DESRES Protein Folding:** Support for D. E. Shaw Research datasets (Lindorff-Larsen et al., 2011).
