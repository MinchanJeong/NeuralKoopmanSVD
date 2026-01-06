# experiments/pipelines/chignolin.py

import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from .base import BasePipeline

# Import Factory
from experiments.factory import get_datasets

# Import optional dependencies
import mdtraj as md
import ase
from ase import Atoms
import dcor

logger = logging.getLogger(__name__)


class ChignolinPipeline(BasePipeline):
    """Pipeline for Chignolin Protein Dynamics experiments.

    Handles the end-to-end workflow for the high-dimensional molecular dynamics benchmark,
    including MDTraj processing, SchNetPack database creation, and physical validation.

    Attributes:
        UNFOLDED_INDICES (range): File indices for unfolded trajectories (0-16).
        FOLDED_INDICES (range): File indices for folded trajectories (17-33).
    """

    UNFOLDED_INDICES = range(17)
    FOLDED_INDICES = range(17, 34)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.raw_path = Path(cfg.data.raw_path)
        self.output_dir = Path(cfg.data.dataset_dir)
        self.topology_file = self.raw_path / "protein.gro"

    def preprocess(self, overwrite: bool = False):
        """Converts raw MD trajectories (XTC/GRO) into SchNetPack databases.

        This process involves:
        1. Filtering trajectories based on mode (full/folded/unfolded).
        2. Splitting into Train/Validation sets.
        3. Computing neighbor lists and saving to SQLite databases (.db).

        Args:
            overwrite (bool): If True, regenerates databases even if they exist.
        """
        if not self.topology_file.exists():
            raise FileNotFoundError(f"Topology file not found: {self.topology_file}")

        mode = getattr(self.cfg.data, "mode", "full")
        split_ratio = self.cfg.data.split[0]
        data_seed = self.cfg.data.seed

        logger.info(
            f"Preprocessing Chignolin: Mode={mode}, Split={split_ratio}, Seed={data_seed}"
        )

        # 1. Prepare File Lists
        unfolded_files = self._get_trajectory_paths("C", self.UNFOLDED_INDICES)
        folded_files = self._get_trajectory_paths("C", self.FOLDED_INDICES)

        if mode == "full":
            target_unfolded, target_folded = unfolded_files, folded_files
        elif mode == "unfolded":
            target_unfolded, target_folded = unfolded_files, []
        elif mode == "folded":
            target_unfolded, target_folded = [], folded_files
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 2. Split Files
        train_u, val_u = self._split_files(target_unfolded, split_ratio, data_seed)
        train_f, val_f = self._split_files(target_folded, split_ratio, data_seed + 1)

        train_files = sorted(train_u + train_f)
        val_files = sorted(val_u + val_f)

        # 3. Create DBs
        split_name = f"split_{int(split_ratio * 100)}_seed{data_seed}"
        db_base_dir = self.output_dir / split_name
        db_base_dir.mkdir(parents=True, exist_ok=True)

        train_db_path = db_base_dir / f"train_{mode}.db"
        val_db_path = db_base_dir / f"val_{mode}.db"

        self._create_db(train_files, train_db_path, "Train", overwrite)
        self._create_db(val_files, val_db_path, "Validation", overwrite)

        # 4. Save Metadata
        metadata = {
            "config": self.cfg.to_dict(),
            "mode": mode,
            "train_files": [f.name for f in train_files],
            "val_files": [f.name for f in val_files],
            "train_db": str(train_db_path),
            "val_db": str(val_db_path),
        }
        with open(db_base_dir / f"metadata_{mode}.json", "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        logger.info(f"Preprocessing Complete. Metadata saved to {db_base_dir}")

    def _get_trajectory_paths(self, prefix: str, indices: range) -> List[Path]:
        files = [self.raw_path / f"{prefix}{i}.xtc" for i in indices]
        existing = [f for f in files if f.exists()]
        return sorted(existing)

    def _split_files(
        self, files: List[Path], ratio: float, seed: int
    ) -> Tuple[List[Path], List[Path]]:
        if not files:
            return [], []
        rng = random.Random(seed)
        shuffled = rng.sample(files, len(files))
        n_train = int(len(shuffled) * ratio)
        return shuffled[:n_train], shuffled[n_train:]

    def _create_db(
        self, file_list: List[Path], db_path: Path, desc: str, overwrite: bool
    ):
        if not file_list:
            return
        if db_path.exists() and not overwrite:
            logger.info(f"{desc} DB exists at {db_path}. Skipping.")
            return
        if db_path.exists():
            db_path.unlink()

        logger.info(f"Creating {desc} DB at {db_path}...")

        # FIX: Use pure ase.db to bypass ASEAtomsData issues entirely during creation.
        # This is more robust against version mismatches.
        total_frames = 0

        try:
            with ase.db.connect(str(db_path), append=False) as conn:
                # 1. Write Metadata first
                conn.metadata = {"_distance_unit": "Ang", "_property_unit_dict": {}}
                # 2. Loop and Write
                for traj_path in tqdm(
                    file_list, desc=f"Processing {desc} data", unit="file"
                ):
                    atoms_list = self._process_single_trajectory(
                        traj_path, self.topology_file
                    )
                    for atoms in atoms_list:
                        # Direct write to SQLite
                        conn.write(atoms)
                        total_frames += 1

        except Exception as e:
            logger.error(f"Failed to create DB: {e}")
            # Clean up partial file
            if db_path.exists():
                db_path.unlink()
            raise

        logger.info(f"Finished {desc} data. Total frames: {total_frames}")

    @staticmethod
    def _process_single_trajectory(
        traj_path: Path, topology_path: Path, chunk_size: int = 1000
    ) -> List["Atoms"]:
        """Loads a trajectory in chunks to optimize memory usage and conversion speed.

        This method leverages `mdtraj.iterload` to stream data, avoiding memory overflow
        on large trajectories. It also utilizes vectorized NumPy operations for unit
        conversion (nm to Å) to minimize Python loop overhead.

        Args:
            traj_path: Path to the trajectory file (.xtc).
            topology_path: Path to the topology file (.gro).
            chunk_size: Number of frames to load per iteration.

        Returns:
            List[Atoms]: A list of ASE Atoms objects corresponding to each frame.
        """
        ase_atoms_list = []

        try:
            # Stream trajectory to prevent memory swapping
            iterator = md.iterload(
                str(traj_path), top=str(topology_path), chunk=chunk_size
            )

            atomic_numbers = None

            for chunk in iterator:
                # Extract topology only once per file to reduce overhead
                if atomic_numbers is None:
                    atomic_numbers = np.array(
                        [atom.element.atomic_number for atom in chunk.topology.atoms]
                    )

                # Vectorized unit conversion (nm -> Å) for the entire chunk
                # Significantly faster than scalar multiplication inside the loop
                xyz_ang = chunk.xyz * 10.0

                has_cell = chunk.unitcell_vectors is not None
                if has_cell:
                    cells_ang = chunk.unitcell_vectors * 10.0

                # Construct ASE Atoms objects
                n_frames = chunk.n_frames
                for i in range(n_frames):
                    # Zero-copy slicing from pre-calculated arrays
                    pos = xyz_ang[i]

                    if has_cell:
                        cell = cells_ang[i]
                        pbc = True
                    else:
                        cell = None
                        pbc = False

                    ase_atoms_list.append(
                        Atoms(
                            positions=pos,
                            numbers=atomic_numbers,
                            cell=cell,
                            pbc=pbc,
                        )
                    )

        except Exception as e:
            logger.error(f"Error processing {traj_path.name}: {e}")
            return []

        return ase_atoms_list

    def analyze(self, run_dir: Path, output_dir: Path):
        """Performs physical validation of the learned Koopman operator.

        Metrics computed:
        - VAMP-2 / VAMP-E Scores: Measure of operator approximation quality.
        - Relaxation Timescales: Inferred from eigenvalues via t = -lag / ln|lambda|.
        - Distance Correlation: Measures alignment between the first eigenmode
          and the physical folding reaction coordinate (RMSD).

        Args:
            run_dir (Path): Path to the experiment results.
            output_dir (Path): Path to save plots and JSON metrics.

        TODO:
            Refactor this method into smaller helpers (_run_inference_loop,
            _compute_physical_metrics, etc.) to improve modularity.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Load Model & Estimator
        logger.info(f"Loading model from {run_dir}")
        pl_module, estimator = self.load_model_and_estimator(run_dir)
        # 2. Setup Data
        train_ds, val_ds, collate_fn = get_datasets(self.cfg)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            pin_memory=True,
        )
        # 3. Load Topology for RMSD calculations
        topo_data = self._load_topology_indices()
        # 4. Run Inference Loop
        inference_results = self._run_inference_loop(
            pl_module, estimator, val_loader, topo_data
        )
        # 5. Compute Metrics & Save Results
        self._compute_and_save_metrics(inference_results, estimator, output_path)

    def _load_topology_indices(self):
        """Loads topology and pre-computes atom indices for RMSD."""
        if md is None:
            return None

        try:
            topo = md.load(str(self.topology_file)).topology
            return {
                "ca_indices": topo.select("name CA"),
                "non_h_indices": topo.select("element != H"),
                "n_atoms": topo.n_atoms,
            }
        except Exception as e:
            logger.warning(f"Failed to load topology: {e}")
            return None

    def _run_inference_loop(self, pl_module, estimator, loader, topo_data):
        """Iterates over validation data to collect modes and calculate RMSDs."""
        logger.info("Running inference on validation set...")
        estimator.reset_eval()

        f_first_modes = []
        rmsd_ca_list = []
        rmsd_nonh_list = []

        # Determine target mode index:
        # If centering is on, Index 0 is constant (1.0), so we want Index 1.
        # If centering is off, the model likely learns the slowest dynamic mode at Index 0.
        target_mode_idx = 1 if self.cfg.model.centering else 0
        device = pl_module.device

        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                x, y = batch

                # Move to device
                x = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }
                y = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in y.items()
                }

                # Forward Pass
                f_x = pl_module(x, lagged=False)
                g_y = pl_module(y, lagged=True)

                # Apply Singular Value Scaling
                if pl_module.svals is not None:
                    scale = pl_module.svals.view(1, -1).sqrt()
                    f_x = f_x * scale
                    g_y = g_y * scale

                f_np = f_x.cpu().numpy()
                g_np = g_y.cpu().numpy()

                # Accumulate Statistics for VAMP-E
                estimator.partial_evaluate(f_np, g_np)

                # Collect Target Mode
                if estimator.use_cca:
                    # Project to canonical basis
                    phi = f_np @ estimator.alignments["f"].T
                else:
                    phi = f_np

                f_first_modes.append(phi[:, target_mode_idx])

                # Calculate RMSDs
                if topo_data:
                    self._compute_batch_rmsd(
                        x, f_x.shape[0], topo_data, rmsd_ca_list, rmsd_nonh_list
                    )

        return {
            "f_first_modes": np.concatenate(f_first_modes) if f_first_modes else None,
            "rmsd_ca": np.concatenate(rmsd_ca_list) if rmsd_ca_list else None,
            "rmsd_nonh": np.concatenate(rmsd_nonh_list) if rmsd_nonh_list else None,
        }

    def _compute_batch_rmsd(self, x, batch_size, topo_data, ca_list, nonh_list):
        """Computes self-centroid RMSD for a batch."""
        pos_tensor = x.get("_positions", x.get("R"))
        if pos_tensor is None:
            return

        try:
            n_atoms = topo_data["n_atoms"]
            pos_np = pos_tensor.cpu().numpy().reshape(batch_size, n_atoms, 3)

            # 1. C-alpha RMSD
            ca_idx = topo_data["ca_indices"]
            pos_ca = pos_np[:, ca_idx, :]
            diff_ca = pos_ca - pos_ca.mean(axis=1, keepdims=True)
            rmsd_ca = np.sqrt(np.mean(np.sum(diff_ca**2, axis=2), axis=1))
            ca_list.append(rmsd_ca)

            # 2. Non-Hydrogen RMSD
            nonh_idx = topo_data["non_h_indices"]
            pos_nonh = pos_np[:, nonh_idx, :]
            diff_nonh = pos_nonh - pos_nonh.mean(axis=1, keepdims=True)
            rmsd_nonh = np.sqrt(np.mean(np.sum(diff_nonh**2, axis=2), axis=1))
            nonh_list.append(rmsd_nonh)

        except Exception as e:
            logger.warning(f"RMSD computation skipped for batch: {e}")

    def _compute_and_save_metrics(self, results, estimator, output_path):
        """Computes scores, timescales, correlations, and saves summary data."""
        # 1. Basic Scores (VAMP-2, VAMP-E)
        scores = estimator.evaluate()
        logger.info(f"Validation Scores: {scores}")

        # 2. Eigenvalues & Timescales
        eigenvalues = None
        timescales = None
        try:
            eigenvalues = estimator.eig()
            valid_mask = np.abs(eigenvalues) > 1e-10
            valid_eigs = np.abs(eigenvalues[valid_mask])

            # Sort descending
            sorted_eigs = np.sort(valid_eigs)[::-1]

            # Identify indices for dynamics (exclude index 0 if centering=True)
            start_idx = 1 if self.cfg.model.centering else 0
            target_eigs = sorted_eigs[start_idx:]

            # Timescale: t = -lag / ln|lambda|
            lag_time_ns = 0.1 * self.cfg.data.time_lag
            timescales = -lag_time_ns / np.log(target_eigs)

            scores["eigenvalues"] = target_eigs.tolist()
            scores["timescales"] = timescales.tolist()
        except Exception as e:
            logger.warning(f"Timescale computation failed: {e}")

        # 3. Gram Matrices (Train & Test)
        # Test Set Gram Matrix (Accumulated during inference loop)
        if estimator._eval_n > 0:
            eval_M_f = estimator._eval_accum["M_f_rho0"] / estimator._eval_n
            eval_M_g = estimator._eval_accum["M_g_rho1"] / estimator._eval_n
        else:
            eval_M_f = None
            eval_M_g = None

        # Train Set Gram Matrix (Directly from Estimator stats)
        # Estimator.stats["raw"] is a NamedTuple (OperatorStats)
        train_stats = estimator.stats.get("raw")
        if train_stats is not None:
            train_M_f = train_stats.M_f
            train_M_g = train_stats.M_g
        else:
            train_M_f = None
            train_M_g = None

        # 4. Correlations
        # Rename for clarity: This is the first DYNAMIC mode (phi_2 if centered)
        first_nontrivial_phi = results["f_first_modes"]
        rmsd_ca = results["rmsd_ca"]
        rmsd_nonh = results["rmsd_nonh"]

        if first_nontrivial_phi is not None and rmsd_ca is not None:
            if dcor is not None:
                # Compute distance correlations
                dcor_ca = float(
                    dcor.distance_correlation(first_nontrivial_phi, rmsd_ca)
                )
                dcor_nonh = float(
                    dcor.distance_correlation(first_nontrivial_phi, rmsd_nonh)
                )

                # Add to scores for metrics.json
                scores["dcor_ca"] = dcor_ca
                scores["dcor_nonh"] = dcor_nonh

                logger.info(f"DistCorr (Ca): {dcor_ca:.4f}")
                logger.info(f"DistCorr (Non-H): {dcor_nonh:.4f}")

            # 5. Save Analysis Data (Summary Only)
            save_dict = {
                "eigenvalues": eigenvalues,
                "timescales": timescales,
                # Save computed dcor values as scalars
                "dcor_ca": scores.get("dcor_ca", np.nan),
                "dcor_nonh": scores.get("dcor_nonh", np.nan),
            }

            # Save Gram Matrices
            if eval_M_f is not None:
                save_dict["gram_f_val"] = eval_M_f
                save_dict["gram_g_val"] = eval_M_g
            if train_M_f is not None:
                save_dict["gram_f_train"] = train_M_f
                save_dict["gram_g_train"] = train_M_g

            np.savez(output_path / "analysis_data.npz", **save_dict)

            # 6. Plots (Use in-memory data)
            self._plot_correlation(
                rmsd_ca,
                first_nontrivial_phi,
                output_path / "correlation_ca.png",
                "Ca RMSD vs First Non-trivial Mode",
            )
            self._plot_correlation(
                rmsd_nonh,
                first_nontrivial_phi,
                output_path / "correlation_nonh.png",
                "Non-H RMSD vs First Non-trivial Mode",
            )

        if timescales is not None:
            self._plot_timescales(timescales, output_path / "timescales.pdf")

        if eval_M_f is not None and train_M_f is not None:
            self._plot_orthogonality(
                train_M_f,
                train_M_g,
                eval_M_f,
                eval_M_g,
                output_path / "orthogonality.pdf",
            )

        # Save Metrics JSON
        with open(output_path / "metrics.json", "w") as f:
            json.dump(scores, f, indent=4)

    def _plot_timescales(self, timescales, save_path):
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(1, len(timescales) + 1)
        ax.plot(
            x,
            timescales,
            "o-",
            color="tab:blue",
            linewidth=1.5,
            markersize=5,
            label="Ours",
        )
        ax.set_title("Estimated Timescale", fontsize=16)
        ax.set_xlabel("Mode Index", fontsize=13)
        ax.set_ylabel("Timescale (ns)", fontsize=13)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 200)
        ax.set_xlim(0, 15)
        ax.grid(True, linestyle="-", alpha=0.8)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_orthogonality(self, M_f_train, M_g_train, M_f_val, M_g_val, save_path):
        def to_corr(M):
            d = np.diag(M)
            d[d < 1e-12] = 1.0
            inv_sqrt = 1.0 / np.sqrt(d)
            D = np.diag(inv_sqrt)
            return D @ M @ D

        fig = plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.08], wspace=0.2, hspace=0.3)

        # Row 1: Train
        for i, (M, label) in enumerate(
            [(M_f_train, r"Train $\mathbf{f}$"), (M_g_train, r"Train $\mathbf{g}$")]
        ):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(to_corr(M), cmap="bwr", vmin=-1, vmax=1)
            ax.set_title(label, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 1:
                plt.colorbar(im, cax=fig.add_subplot(gs[0, 2]))

        # Row 2: Test
        for i, (M, label) in enumerate(
            [(M_f_val, r"Test $\mathbf{f}$"), (M_g_val, r"Test $\mathbf{g}$")]
        ):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(to_corr(M), cmap="bwr", vmin=-1, vmax=1)
            ax.set_title(label, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 1:
                plt.colorbar(im, cax=fig.add_subplot(gs[1, 2]))

        plt.suptitle("Orthogonality of Basis Functions", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_correlation(self, rmsd, phi, save_path, title):
        plt.figure(figsize=(6, 5))
        plt.scatter(rmsd, phi, alpha=0.1, s=1, c="tab:blue")
        plt.xlabel("RMSD (Å)", fontsize=12)
        plt.ylabel(r"First Non-Trivial Mode ($\phi$)", fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
