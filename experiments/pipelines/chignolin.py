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

# Conditional imports
try:
    import mdtraj as md
    from ase import Atoms
    from schnetpack.data import ASEAtomsData
except ImportError:
    md = None
    Atoms = None
    ASEAtomsData = None

try:
    import dcor
except ImportError:
    dcor = None

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

        if md is None or ASEAtomsData is None:
            raise ImportError(
                "Chignolin pipeline requires 'mdtraj', 'ase', 'scipy', and 'schnetpack'."
            )

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
        db_conn = ASEAtomsData.create(
            str(db_path), distance_unit="Ang", property_unit_dict={}
        )

        total_frames = 0
        for traj_path in tqdm(file_list, desc=f"Processing {desc}", unit="file"):
            atoms_list = self._process_single_trajectory(traj_path, self.topology_file)
            if atoms_list:
                properties_list = [{} for _ in range(len(atoms_list))]
                try:
                    db_conn.add_systems(
                        atoms_list=atoms_list, property_list=properties_list
                    )
                    total_frames += len(atoms_list)
                except Exception as e:
                    logger.error(f"Failed to add system from {traj_path.name}: {e}")
        logger.info(f"Finished {desc}. Total frames: {total_frames}")

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

        # 3. Topology for RMSD
        if md is not None:
            topo = md.load(str(self.topology_file)).topology
            ca_indices = topo.select("name CA")
            n_atoms = topo.n_atoms

        # 4. Inference Loop
        logger.info("Running inference on validation set...")
        estimator.reset_eval()

        f_first_modes = []
        rmsd_list = []
        target_mode_idx = 1 if self.cfg.model.centering else 0
        device = pl_module.device

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Eval"):
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

                # Apply scaling if learned (Important for VAMP calculation)
                if pl_module.svals is not None:
                    scale = pl_module.svals.view(1, -1).sqrt()
                    f_x = f_x * scale
                    g_y = g_y * scale

                f_np = f_x.cpu().numpy()
                g_np = g_y.cpu().numpy()

                # Accumulate Stats for VAMP-E
                estimator.partial_evaluate(f_np, g_np)

                # Compute RMSD & Extract Mode 1
                if md is not None:
                    pos_tensor = x.get("_positions", x.get("R"))
                    if pos_tensor is not None:
                        try:
                            pos_np = (
                                pos_tensor.cpu()
                                .numpy()
                                .reshape(f_x.shape[0], n_atoms, 3)
                            )
                            pos_ca = pos_np[:, ca_indices, :]
                            diff = pos_ca - pos_ca.mean(axis=1, keepdims=True)
                            batch_rmsds = np.sqrt(
                                np.mean(np.sum(diff**2, axis=2), axis=1)
                            )
                            rmsd_list.append(batch_rmsds)
                        except Exception:
                            pass

                # Project to aligned basis for correlation analysis
                if estimator.use_cca:
                    phi = f_np @ estimator.alignments["f"].T
                else:
                    phi = f_np
                f_first_modes.append(phi[:, target_mode_idx])

        # 5. Compute Metrics
        scores = estimator.evaluate()  # Computes VAMP-2 and VAMP-E using Training Sigma
        logger.info(f"Validation Scores: {scores}")

        eigenvalues = None
        timescales = None

        try:
            eigenvalues = estimator.eig()
            valid_mask = np.abs(eigenvalues) > 1e-10
            valid_eigs = np.abs(eigenvalues[valid_mask])

            # Sort and select
            start_idx = 1 if self.cfg.model.centering else 0
            sorted_eigs = np.sort(valid_eigs)[::-1]
            target_eigs = sorted_eigs[start_idx:]

            # Timescale: t = -lag / ln|lambda|
            lag_time_ns = 0.1 * self.cfg.data.time_lag
            timescales = -lag_time_ns / np.log(target_eigs)

            scores["eigenvalues"] = target_eigs.tolist()
            scores["timescales"] = timescales.tolist()
        except Exception as e:
            logger.warning(f"Timescale computation failed: {e}")

        # 6. Save & Plot
        if rmsd_list and f_first_modes:
            full_rmsd = np.concatenate(rmsd_list)
            full_phi1 = np.concatenate(f_first_modes)

            if dcor is not None:
                scores["dist_corr_ca"] = float(
                    dcor.distance_correlation(full_phi1, full_rmsd)
                )
                logger.info(f"Distance Correlation: {scores['dist_corr_ca']:.4f}")

            np.savez(
                output_path / "analysis_data.npz",
                rmsd=full_rmsd,
                phi1=full_phi1,
                eigenvalues=eigenvalues,
                timescales=timescales,
            )
            self._plot_correlation(
                full_rmsd, full_phi1, output_path / "correlation.png"
            )

        with open(output_path / "metrics.json", "w") as f:
            json.dump(scores, f, indent=4)

        if timescales is not None:
            self._plot_timescales(timescales, output_path / "timescales.pdf")

        # Plot Orthogonality using accumulated test stats
        eval_M_f = estimator._eval_accum["M_f_rho0"]
        eval_M_g = estimator._eval_accum["M_g_rho1"]
        if eval_M_f is not None:
            self._plot_orthogonality(
                eval_M_f, eval_M_g, output_path / "orthogonality.pdf"
            )

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
        ax.set_xlabel("absolute eigenvalue index", fontsize=13)
        ax.set_ylabel("timescale (ns)", fontsize=13)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 200)
        ax.set_xlim(0, 15)
        xticks = np.arange(1, len(timescales) + 1, 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        lag_time_ns = 0.1 * self.cfg.data.time_lag
        ax.hlines(lag_time_ns, 0, 16, color="k", linestyle="--", linewidth=1.5)
        ax.grid(True, linestyle="-", alpha=0.8)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_orthogonality(self, M_f, M_g, save_path):
        def to_corr(M):
            d = np.diag(M)
            inv_sqrt = 1.0 / np.sqrt(d)
            inv_sqrt[np.isinf(inv_sqrt)] = 0.0
            D = np.diag(inv_sqrt)
            return D @ M @ D

        fig = plt.figure(figsize=(7, 3))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.2)
        ax_f = fig.add_subplot(gs[0, 0])
        ax_f.imshow(to_corr(M_f), cmap="bwr", vmin=-1, vmax=1)
        ax_f.set_title(r"$\mathbf{f}$", fontsize=14)
        ax_f.set_xticks([])
        ax_f.set_yticks([])

        ax_g = fig.add_subplot(gs[0, 1])
        im = ax_g.imshow(to_corr(M_g), cmap="bwr", vmin=-1, vmax=1)
        ax_g.set_title(r"$\mathbf{g}$", fontsize=14)
        ax_g.set_xticks([])
        ax_g.set_yticks([])

        cax = fig.add_subplot(gs[0, 2])
        fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def _plot_correlation(self, rmsd, phi1, save_path):
        plt.figure(figsize=(6, 5))
        plt.scatter(rmsd, phi1, alpha=0.1, s=1, c="tab:blue")
        plt.xlabel("RMSD (Å)", fontsize=12)
        plt.ylabel(r"First Eigenmode ($\phi_1$)", fontsize=12)
        plt.title("Correlation: Structure vs Dynamics", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
