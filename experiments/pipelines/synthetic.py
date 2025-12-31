# experiments/pipelines/synthetic.py
import logging
import numpy as np
import json
from pathlib import Path

from .base import BasePipeline
from koopmansvd.systems import LogisticMap
from koopmansvd.models.metrics import directed_hausdorff_distance

logger = logging.getLogger(__name__)


class SyntheticPipeline(BasePipeline):
    """
    Pipeline for Synthetic Datasets (Logistic, Lorenz, etc.)

    - Preprocess: Generates trajectory data on-the-fly and saves to .npy
    - Analyze: Compares learned eigenvalues with Ground Truth (if available)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset_name = cfg.data.dataset_name
        self.save_path = Path(cfg.data.path)

    def _get_generator(self):
        # Initialize generator based on configuration
        if self.dataset_name == "logistic":
            return LogisticMap(r=4.0, N=20, rng_seed=self.cfg.data.seed)
        else:
            raise ValueError(f"Unknown synthetic dataset: {self.dataset_name}")

    def preprocess(self, overwrite=False):
        """Generates data and saves to .npy"""
        if self.save_path.exists() and not overwrite:
            logger.info(
                f"Data already exists at {self.save_path}. Skipping generation."
            )
            return

        logger.info(f"Generating synthetic data: {self.dataset_name}...")
        generator = self._get_generator()

        n_samples = getattr(self.cfg.data, "n_samples", 50000)

        x0_dim = 1 if self.dataset_name in ["logistic"] else 3

        # Initialize state (random)
        x0 = np.random.randn(x0_dim)

        # Generate data
        traj = generator.sample(x0, T=n_samples)

        # Save (for TensorContextDataset compatibility)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.save_path, traj)
        logger.info(f"Saved trajectory shape {traj.shape} to {self.save_path}")

    def analyze(self, run_dir: Path, output_dir: Path):
        """Compares learned eigenvalues with Ground Truth"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        _, estimator = self.load_model_and_estimator(run_dir)
        if estimator is None:
            logger.warning("Estimator could not be loaded. Skipping analysis.")
            return

        # Compute eigenvalues from the restored operator
        learned_eigs = estimator.eig()

        generator = self._get_generator()

        # Get GT if available
        # Note: generator.eig() might need implementation or return None if expensive
        gt_eigs = None
        if hasattr(generator, "eig"):
            gt_eigs = generator.eig()

        if gt_eigs is None:
            logger.info("Ground Truth Eigenvalues not available.")
            return

        logger.info("Computing Analysis Metrics...")

        if self.dataset_name == "logistic":
            # Hausdorff on complex plane
            h_dist = directed_hausdorff_distance(learned_eigs, gt_eigs)
            logger.info(f"Directed Hausdorff Distance: {h_dist:.6f}")

            with open(output_path / "metrics.json", "w") as f:
                json.dump({"hausdorff": float(h_dist)}, f)
