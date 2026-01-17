# experiments/pipelines/chignolin.py

import logging
import json
import numpy as np
import scipy
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
        self.rmsd_cache_file = self.output_dir / "rmsd_full_cache.npz"

    # --- Stage 1: Preprocessing --
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

        self.output_dir.mkdir(parents=True, exist_ok=True)
        mode = getattr(self.cfg.data, "mode", "full")
        target_split_ratio = self.cfg.data.split[0]

        # 1. Prepare File Lists
        unfolded_files = self._get_trajectory_paths("C", self.UNFOLDED_INDICES)
        folded_files = self._get_trajectory_paths("C", self.FOLDED_INDICES)

        # Determine target files based on mode
        if mode == "full":
            target_unfolded, target_folded = unfolded_files, folded_files
        elif mode == "unfolded":
            target_unfolded, target_folded = unfolded_files, []
        elif mode == "folded":
            target_unfolded, target_folded = [], folded_files
        else:
            raise ValueError(f"Invalid mode: {mode}")

        all_target_files = sorted(target_unfolded + target_folded)

        # 2. Cache RMSD (For eval and distribution matching)
        # We perform this if cache is missing OR if we are in distribution_matching mode
        # (to ensure we have data to optimize).
        rmsd_cache = {}
        if not self.rmsd_cache_file.exists() or overwrite:
            self._cache_rmsds(all_target_files)

        # Load cache into memory to pass to _create_db later
        # (Avoids re-calculation during DB writing)
        raw_cache = np.load(self.rmsd_cache_file)
        # Reconstruct dict: {filename: {'rmsd_ca': ..., 'rmsd_nonh': ...}}
        for key in raw_cache.files:
            # key format: "filename_type" (e.g., "C0.xtc_ca")
            fname, metric = key.rsplit("_", 1)
            if fname not in rmsd_cache:
                rmsd_cache[fname] = {}
            rmsd_cache[fname][f"rmsd_{metric}"] = raw_cache[key]

        # 3. Determine Split (Seed & Files)
        split_strategy = getattr(self.cfg.data, "split_strategy", "random")
        search_seed = self.cfg.data.seed
        if split_strategy == "distribution_matching":
            assert mode == "full", (
                "Distribution matching split only supported in 'full' mode."
            )
            logger.info(
                "[Strategy] Distribution Matching: Optimizing split for thermodynamic consistency."
            )

            # Find Golden Seed using Wasserstein Distance
            # Note: We enforce the found seed here, overriding cfg.data.seed for data splitting
            best_seed, train_files, val_files = self._find_golden_split(
                target_unfolded,
                target_folded,
                rmsd_cache,
                target_split_ratio,
                search_seed,
                n_trials=100,
            )
            data_seed = best_seed

        else:
            # Default: Random Strategy
            data_seed = search_seed
            logger.info(f"[Strategy] Random: Splitting using fixed seed {data_seed}.")

            train_u, val_u = self._split_files(
                target_unfolded, target_split_ratio, data_seed
            )
            train_f, val_f = self._split_files(
                target_folded, target_split_ratio, data_seed + 1
            )

            train_files = sorted(train_u + train_f)
            val_files = sorted(val_u + val_f)

        # 4. Create DBs (Using Cached RMSD)
        split_name = f"split_{int(target_split_ratio * 100)}_seed{data_seed}"
        db_base_dir = self.output_dir / split_name
        db_base_dir.mkdir(parents=True, exist_ok=True)

        train_db_path = db_base_dir / f"train_{mode}.db"
        val_db_path = db_base_dir / f"val_{mode}.db"

        # Pass rmsd_cache to _create_db to avoid re-calculation
        self._create_db(train_files, train_db_path, "Train", overwrite, rmsd_cache)
        self._create_db(val_files, val_db_path, "Validation", overwrite, rmsd_cache)

        # 4. Save Metadata
        metadata = {
            "config": self.cfg.to_dict(),
            "mode": mode,
            "split_strategy": split_strategy,
            "split_ratio": target_split_ratio,
            "search_seed": int(search_seed),
            "effective_data_seed": int(data_seed),
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

    def _cache_rmsds(self, file_list: List[Path]):
        """
        Reads all trajectories once and saves BOTH CA and Non-H RMSD.
        """
        logger.info("Caching RMSD data (CA & Non-H) for efficiency...")
        topo_data = self._load_topology_indices()
        cache_data = {}

        for traj_path in tqdm(file_list, desc="Caching RMSD"):
            # Compute both metrics
            result = self._compute_rmsd_values(traj_path, self.topology_file, topo_data)
            # Flatten keys for npz saving: "filename_ca"
            cache_data[f"{traj_path.name}_ca"] = result["rmsd_ca"]

        np.savez(self.rmsd_cache_file, **cache_data)
        logger.info(f"Full RMSD cache saved to {self.rmsd_cache_file}")

    def _compute_rmsd_values(
        self, traj_path, topology_path, topo_data
    ) -> Dict[str, np.ndarray]:
        """Computes RMSD."""
        rmsds_ca = []
        rmsds_nonh = []

        ca_idx = topo_data["ca_indices"]
        nonh_idx = topo_data["non_h_indices"]
        n_atoms = topo_data["n_atoms"]

        try:
            for chunk in md.iterload(
                str(traj_path), top=str(topology_path), chunk=1000
            ):
                xyz_ang = chunk.xyz * 10.0
                pos_np = xyz_ang.reshape(chunk.n_frames, n_atoms, 3)

                # CA RMSD
                pos_ca = pos_np[:, ca_idx, :]
                diff_ca = pos_ca - pos_ca.mean(axis=1, keepdims=True)
                rmsds_ca.append(np.sqrt(np.mean(np.sum(diff_ca**2, axis=2), axis=1)))

                # Non-H RMSD
                pos_nonh = pos_np[:, nonh_idx, :]
                diff_nonh = pos_nonh - pos_nonh.mean(axis=1, keepdims=True)
                rmsds_nonh.append(
                    np.sqrt(np.mean(np.sum(diff_nonh**2, axis=2), axis=1))
                )

            return {
                "rmsd_ca": np.concatenate(rmsds_ca).astype(np.float32),
                "rmsd_nonh": np.concatenate(rmsds_nonh).astype(np.float32),
            }
        except Exception as e:
            logger.error(f"Error computing RMSD for {traj_path}: {e}")
            return {"rmsd_ca": np.array([]), "rmsd_nonh": np.array([])}

    def _find_golden_split(
        self,
        unfolded_files: List[Path],
        folded_files: List[Path],
        rmsd_cache: Dict,
        ratio: float,
        seed: int,
        n_trials: int = 100,
    ):
        logger.info(
            f"Searching for Golden Seed (Distributional Consistency) over {n_trials} trials..."
        )

        best_seed = -1
        min_dist = float("inf")
        best_train = []
        best_val = []

        rng = random.Random(seed)
        # 0 is included to check the default case
        candidate_seeds = [0] + rng.sample(range(1, 100000), n_trials)

        for current_seed in tqdm(candidate_seeds, desc="Optimizing Split"):
            train_u, val_u = self._split_files(unfolded_files, ratio, current_seed)
            train_f, val_f = self._split_files(folded_files, ratio, current_seed + 1)

            train_files = train_u + train_f
            val_files = val_u + val_f

            try:
                # Use cached CA RMSD for distribution matching
                train_dist = np.concatenate(
                    [rmsd_cache[f.name]["rmsd_ca"] for f in train_files]
                )
                val_dist = np.concatenate(
                    [rmsd_cache[f.name]["rmsd_ca"] for f in val_files]
                )
            except KeyError:
                continue

            dist = scipy.stats.wasserstein_distance(train_dist, val_dist)

            if dist < min_dist:
                tqdm.write(f"[Seed {current_seed:05d}] Dist: {dist:.6f} ★ NEW BEST")
                min_dist = dist
                best_seed = current_seed
                best_train = train_files
                best_val = val_files

        logger.info("-" * 50)
        logger.info(
            f"Final Selection -> Seed: {best_seed} (Wasserstein Dist: {min_dist:.6f})"
        )

        def get_indices(file_list):
            return sorted(
                [int("".join(filter(str.isdigit, f.name))) for f in file_list]
            )

        train_indices = get_indices(best_train)
        val_indices = get_indices(best_val)

        logger.info(f"Train Files ({len(train_indices)}): {train_indices}")
        logger.info(f"Val Files   ({len(val_indices)}): {val_indices}")
        logger.info("-" * 50)

        return best_seed, best_train, best_val

    def _create_db(
        self,
        file_list: List[Path],
        db_path: Path,
        desc: str,
        overwrite: bool,
        rmsd_cache: Optional[Dict] = None,
    ):
        if not file_list:
            return
        if db_path.exists() and not overwrite:
            logger.info(f"{desc} DB exists at {db_path}. Skipping.")
            return
        if db_path.exists():
            db_path.unlink()

        logger.info(f"Creating {desc} DB at {db_path}...")

        # If cache is not provided, we must load topology to compute
        topo_data = self._load_topology_indices() if rmsd_cache is None else None

        total_frames = 0
        traj_lengths = []
        try:
            with ase.db.connect(str(db_path), append=False) as conn:
                conn.metadata = {
                    "_distance_unit": "Ang",
                    "_property_unit_dict": {"rmsd_ca": "Ang", "rmsd_nonh": "Ang"},
                }
                for traj_path in tqdm(file_list, desc=f"Writing {desc} DB"):
                    # Retrieve cached RMSD if available
                    precomputed_rmsd = (
                        rmsd_cache.get(traj_path.name) if rmsd_cache else None
                    )

                    pairs_list = self._process_single_trajectory(
                        traj_path,
                        self.topology_file,
                        topo_data=topo_data,
                        precomputed_rmsd=precomputed_rmsd,
                    )

                    if pairs_list:
                        traj_lengths.append(len(pairs_list))
                        for atoms, extra_data in pairs_list:
                            conn.write(atoms, data=extra_data)
                            total_frames += 1

        except Exception as e:
            logger.error(f"Failed to create DB: {e}")
            if db_path.exists():
                db_path.unlink()
            raise

        info_path = db_path.with_suffix(".info.json")
        with open(info_path, "w") as f:
            json.dump({"traj_lengths": traj_lengths}, f, indent=4)

        logger.info(f"Finished {desc} data. Total frames: {total_frames}")

    @staticmethod
    def _process_single_trajectory(
        traj_path: Path,
        topology_path: Path,
        topo_data: Optional[dict] = None,
        chunk_size: int = 1000,
        precomputed_rmsd: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Tuple["Atoms", dict]]:
        data_pairs = []  # Save (atoms, data_dict) tuple

        try:
            iterator = md.iterload(
                str(traj_path), top=str(topology_path), chunk=chunk_size
            )
            atomic_numbers = None
            global_frame_idx = 0

            for chunk in iterator:
                if atomic_numbers is None:
                    atomic_numbers = np.array(
                        [atom.element.atomic_number for atom in chunk.topology.atoms]
                    )

                xyz_ang = chunk.xyz * 10.0
                has_cell = chunk.unitcell_vectors is not None
                cells_ang = chunk.unitcell_vectors * 10.0 if has_cell else None

                n_frames = chunk.n_frames

                # Get RMSD: Either from Cache or Compute on the fly
                if precomputed_rmsd is not None:
                    rmsd_ca_chunk = precomputed_rmsd["rmsd_ca"][
                        global_frame_idx : global_frame_idx + n_frames
                    ]
                    rmsd_nonh_chunk = precomputed_rmsd["rmsd_nonh"][
                        global_frame_idx : global_frame_idx + n_frames
                    ]
                else:
                    # Fallback logic (Only if cache missing)
                    pos_np = xyz_ang.reshape(n_frames, topo_data["n_atoms"], 3)
                    ca_idx, nonh_idx = (
                        topo_data["ca_indices"],
                        topo_data["non_h_indices"],
                    )

                    pos_ca = pos_np[:, ca_idx, :]
                    diff_ca = pos_ca - pos_ca.mean(axis=1, keepdims=True)
                    rmsd_ca_chunk = np.sqrt(np.mean(np.sum(diff_ca**2, axis=2), axis=1))

                    pos_nonh = pos_np[:, nonh_idx, :]
                    diff_nonh = pos_nonh - pos_nonh.mean(axis=1, keepdims=True)
                    rmsd_nonh_chunk = np.sqrt(
                        np.mean(np.sum(diff_nonh**2, axis=2), axis=1)
                    )

                for i in range(n_frames):
                    atoms = Atoms(
                        positions=xyz_ang[i],
                        numbers=atomic_numbers,
                        cell=cells_ang[i] if has_cell else None,
                        pbc=has_cell,
                    )
                    # Convert to float32 explicitly
                    extra_data = {
                        "rmsd_ca": np.array([rmsd_ca_chunk[i]], dtype=np.float32),
                        "rmsd_nonh": np.array([rmsd_nonh_chunk[i]], dtype=np.float32),
                    }
                    data_pairs.append((atoms, extra_data))

                global_frame_idx += n_frames

        except Exception as e:
            logger.error(f"Error processing {traj_path.name}: {e}")
            return []
        return data_pairs

    def _load_topology_indices(self):
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

    # --- Stage 2: Analysis ---
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
        """

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting analysis for run: {run_dir}")

        pl_module, estimator = self.load_model_and_estimator(run_dir)
        if pl_module is None or estimator is None:
            logger.error("Model or estimator failed to load. Aborting analysis.")
            return

        _, val_ds, collate_fn = get_datasets(self.cfg)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
        )

        raw_data_paths = self._run_inference_and_collect_data(
            pl_module, estimator, val_loader, output_dir, len(val_ds)
        )
        analysis_results = self._compute_and_save_analysis(
            estimator, raw_data_paths, output_dir
        )
        self._generate_plots(output_dir, analysis_results)

        logger.info(f"Analysis complete. Results saved in {output_dir}")

    def _run_inference_and_collect_data(
        self, pl_module, estimator, loader, output_dir, n_samples
    ):
        logger.info("Stage 1: Running inference and collecting raw data...")
        estimator.reset_eval()
        device = pl_module.device
        target_mode_idx = 1 if self.cfg.model.centering else 0

        phi_path = output_dir / "modes_raw.mmap"
        rmsd_ca_path = output_dir / "rmsd_ca_raw.mmap"
        rmsd_nonh_path = output_dir / "rmsd_nonh_raw.mmap"

        phi_mmap = np.memmap(phi_path, dtype=np.float32, mode="w+", shape=(n_samples,))
        rmsd_ca_mmap = np.memmap(
            rmsd_ca_path, dtype=np.float32, mode="w+", shape=(n_samples,)
        )
        rmsd_nonh_mmap = np.memmap(
            rmsd_nonh_path, dtype=np.float32, mode="w+", shape=(n_samples,)
        )

        current_idx = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                x, y = batch
                x = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }
                y = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in y.items()
                }

                f_x, g_y = pl_module(x, False), pl_module(y, True)

                if pl_module.svals is not None:
                    scale = pl_module.svals.view(1, -1).sqrt()
                    f_x, g_y = f_x * scale, g_y * scale

                f_np, g_np = f_x.cpu().numpy(), g_y.cpu().numpy()
                estimator.partial_evaluate(f_np, g_np)
                phi = f_np @ estimator.alignments["f"].T if estimator.use_cca else f_np

                batch_size = f_np.shape[0]
                end_idx = current_idx + batch_size

                phi_mmap[current_idx:end_idx] = phi[:, target_mode_idx]
                rmsd_ca_mmap[current_idx:end_idx] = x["rmsd_ca"].cpu().numpy().flatten()
                rmsd_nonh_mmap[current_idx:end_idx] = (
                    x["rmsd_nonh"].cpu().numpy().flatten()
                )
                current_idx = end_idx

        phi_mmap.flush()
        rmsd_ca_mmap.flush()
        rmsd_nonh_mmap.flush()

        return {"phi": phi_path, "rmsd_ca": rmsd_ca_path, "rmsd_nonh": rmsd_nonh_path}

    def _compute_and_save_analysis(self, estimator, data_paths, output_dir):
        logger.info("Stage 2: Computing and saving analysis metrics...")

        if estimator._eval_n > 0:
            eval_M_f = estimator._eval_accum["M_f_rho0"] / estimator._eval_n
            eval_M_g = estimator._eval_accum["M_g_rho1"] / estimator._eval_n
        else:
            eval_M_f, eval_M_g = None, None

        scores = estimator.evaluate()

        try:
            eigenvalues = estimator.eig()
            valid_eigs = np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10])
            sorted_eigs = np.sort(valid_eigs)[::-1]
            start_idx = 1 if self.cfg.model.centering else 0
            target_eigs = sorted_eigs[start_idx:]

            lag_time_ns = 0.1 * self.cfg.data.time_lag
            timescales = -lag_time_ns / np.log(target_eigs)

            scores.update(
                {"eigenvalues": target_eigs.tolist(), "timescales": timescales.tolist()}
            )
        except Exception as e:
            logger.warning(f"Timescale computation failed: {e}")
            timescales, target_eigs = None, None

        phi = np.memmap(data_paths["phi"], dtype=np.float32, mode="r")
        rmsd_ca = np.memmap(data_paths["rmsd_ca"], dtype=np.float32, mode="r")
        rmsd_nonh = np.memmap(data_paths["rmsd_nonh"], dtype=np.float32, mode="r")

        dcor_ca = float(dcor.distance_correlation(phi, rmsd_ca))
        dcor_nonh = float(dcor.distance_correlation(phi, rmsd_nonh))
        scores.update({"dcor_ca": dcor_ca, "dcor_nonh": dcor_nonh})
        logger.info(f"DistCorr (Ca): {dcor_ca:.4f}, (Non-H): {dcor_nonh:.4f}")

        train_stats = estimator.stats.get("raw")
        train_M_f = train_stats.M_f if train_stats else None
        train_M_g = train_stats.M_g if train_stats else None

        analysis_data = {
            "eigenvalues": target_eigs,
            "timescales": timescales,
            "dcor_ca": dcor_ca,
            "dcor_nonh": dcor_nonh,
            "gram_f_train": train_M_f,
            "gram_g_train": train_M_g,
            "gram_f_val": eval_M_f,
            "gram_g_val": eval_M_g,
        }
        np.savez(output_dir / "analysis_data.npz", **analysis_data)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(scores, f, indent=4)

        return analysis_data

    def _generate_plots(self, output_dir, analysis_results):
        logger.info("Stage 3: Generating plots...")

        if analysis_results["timescales"] is not None:
            self._plot_timescales(
                analysis_results["timescales"], output_dir / "timescales.pdf"
            )

        if (
            analysis_results["gram_f_train"] is not None
            and analysis_results["gram_f_val"] is not None
        ):
            self._plot_orthogonality(
                analysis_results["gram_f_train"],
                analysis_results["gram_g_train"],
                analysis_results["gram_f_val"],
                analysis_results["gram_g_val"],
                output_dir / "orthogonality.pdf",
            )

        phi = np.memmap(output_dir / "modes_raw.mmap", dtype=np.float32, mode="r")
        rmsd_ca = np.memmap(output_dir / "rmsd_ca_raw.mmap", dtype=np.float32, mode="r")
        rmsd_nonh = np.memmap(
            output_dir / "rmsd_nonh_raw.mmap", dtype=np.float32, mode="r"
        )

        self._plot_correlation(
            rmsd_ca, phi, output_dir / "correlation_ca.png", "Ca RMSD vs Mode"
        )
        self._plot_correlation(
            rmsd_nonh, phi, output_dir / "correlation_nonh.png", "Non-H RMSD vs Mode"
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
            d = np.diag(M).copy()
            d[d < 1e-12] = 1.0
            D = np.diag(1.0 / np.sqrt(d))
            return D @ M @ D

        fig = plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.08], wspace=0.2, hspace=0.3)

        plot_data = [
            (M_f_train, r"Train $\mathbf{f}$"),
            (M_g_train, r"Train $\mathbf{g}$"),
            (M_f_val, r"Test $\mathbf{f}$"),
            (M_g_val, r"Test $\mathbf{g}$"),
        ]

        for i, (M, label) in enumerate(plot_data):
            row, col = i // 2, i % 2
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(to_corr(M), cmap="bwr", vmin=-1, vmax=1)
            ax.set_title(label, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 1:
                plt.colorbar(im, cax=fig.add_subplot(gs[row, 2]))

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
