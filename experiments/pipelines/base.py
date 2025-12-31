# experiments/pipelines/base.py

import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path

from experiments.factory import get_encoder, get_loss, get_lightning_module
from koopmansvd.models.inference import KoopmanEstimator

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Abstract base class for experimental pipelines.

    This class defines the standard interface for preprocessing raw data and
    running post-training analysis. Concrete implementations should handle
    dataset-specific logic.

    Attributes:
        cfg (ml_collections.ConfigDict): The experiment configuration object.
    """

    def __init__(self, cfg):
        """Initializes the pipeline.

        Args:
            cfg: The experiment configuration object (ml_collections.ConfigDict).
        """
        self.cfg = cfg

    @abstractmethod
    def preprocess(self, **kwargs):
        """Runs data preprocessing steps (Raw -> Database/Tensor).

        This method should handle loading raw data, splitting it (train/val),
        and saving it in a format efficient for the DataLoader (e.g., .npy or SQLite).

        Args:
            **kwargs: Additional arguments for preprocessing options (e.g., overwrite flags).
        """
        pass

    @abstractmethod
    def analyze(self, run_dir: Path, output_dir: Path):
        """Runs post-training analysis and metric computation.

        Args:
            run_dir (Path): The directory containing the training run artifacts
                (checkpoints, config.yaml).
            output_dir (Path): The directory where analysis results (plots, metrics)
                should be saved.
        """
        pass

    def load_model_and_estimator(self, run_dir: Path):
        """Loads the PLModule and KoopmanEstimator from a run directory.

        Args:
            run_dir (Path): The experiment run directory.

        Returns:
            Tuple[KoopmanPLModule, KoopmanEstimator]: The loaded model and estimator.
            Returns (None, None) if loading fails.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Find Checkpoint
        ckpt_files = list((run_dir / "checkpoints").glob("*.ckpt"))
        if not ckpt_files:
            logger.error(f"No checkpoint found in {run_dir}/checkpoints")
            return None, None
        ckpt_path = ckpt_files[0]
        logger.info(f"Loading checkpoint from: {ckpt_path}")

        # 2. Reconstruct Model Components via Factory
        encoder = get_encoder(self.cfg)
        lagged_encoder = get_encoder(self.cfg)
        loss_fn = get_loss(self.cfg)

        # 3. Load PLModule
        try:
            # Instantiate a blank module with correct structure (handles Langevin potential_fn)
            model = get_lightning_module(self.cfg, encoder, lagged_encoder, loss_fn)

            # Load state dict
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

            # Explicitly move model to device and set to eval mode
            model.to(device)
            model.eval()

        except Exception as e:
            logger.error(f"Failed to load PLModule: {e}")
            return None, None

        # 4. Load Estimator (Full Restore)
        est_path = run_dir / "results.npz"
        if not est_path.exists():
            logger.warning("results.npz not found. Returning model only.")
            return model, None

        try:
            data = np.load(est_path, allow_pickle=True)
            estimator = KoopmanEstimator(use_cca=True)

            # Restore state dicts completely
            estimator.operators = data["operators"].item()
            estimator.stats = data["stats"].item()
            estimator.alignments = data["alignments"].item()
            estimator.projections = data["projections"].item()
            estimator._cca_components = data["cca_components"].item()
            estimator._is_fitted = True

            logger.info("Restored KoopmanEstimator from results.npz")
            return model, estimator

        except Exception as e:
            logger.error(f"Failed to load estimator: {e}")
            return model, None
