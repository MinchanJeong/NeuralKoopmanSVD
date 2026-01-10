# experiments/run_post_inference.py

import os
import sys

sys.path.append(os.getcwd())

import logging
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from ml_collections import ConfigDict

from experiments.factory import (
    get_encoder,
    get_lightning_module,
    get_loss,
    get_datasets,
)
from koopmansvd.models.inference import KoopmanEstimator
from koopmansvd.data.utils import extract_raw_data

logger = logging.getLogger(__name__)


def run_post_training_inference(run_dir: Path, device: torch.device = None):
    """
    Loads the trained model checkpoint and re-computes operator statistics on the full dataset.

    This ensures that the 'results.npz' file contains statistics derived strictly from
    the saved checkpoint weights (globally synchronized) rather than the potentially
    unsynchronized in-memory state remaining after DDP training.

    Args:
        run_dir (Path): The experiment directory containing 'config.yaml' and 'checkpoints/'.
        device (torch.device, optional): Device to run inference on. Defaults to CUDA if available.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"

    # 1. Load Configuration
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")

    with open(config_path, "r") as f:
        cfg = ConfigDict(yaml.unsafe_load(f))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting Post-Training Inference for: {run_dir}")

    # 2. Resolve Checkpoint (Prioritize 'last.ckpt')
    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"

    if not last_ckpt.exists():
        # Fallback: Sort by modification time or name to find the latest
        ckpts = sorted(list(ckpt_dir.glob("*.ckpt")))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        last_ckpt = ckpts[-1]

    logger.info(f"Loading checkpoint: {last_ckpt}")

    # 3. Reconstruct Model & Data Pipeline
    encoder = get_encoder(cfg)
    lagged_encoder = get_encoder(cfg)
    loss_fn = get_loss(cfg)
    pl_module = get_lightning_module(cfg, encoder, lagged_encoder, loss_fn)

    # Load weights from disk to ensure consistency (Fixes DDP buffer mismatch)
    # weights_only=False is required for PyTorch Lightning checkpoints
    checkpoint = torch.load(last_ckpt, map_location=device, weights_only=False)

    print(f"Global Step: {checkpoint['global_step']}")
    print(f"Current Epoch: {checkpoint['epoch']}")

    pl_module.load_state_dict(checkpoint["state_dict"])
    pl_module.to(device)
    pl_module.eval()

    # Load the full training dataset
    train_ds, _, collate_fn = get_datasets(cfg)

    # Configure Loader: Sequential sampling, no dropping
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "prefetch_factor": getattr(cfg.data, "prefetch_factor", 2)
        if cfg.data.num_workers > 0
        else None,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "shuffle": False,
        "collate_fn": collate_fn,
    }
    inference_loader = DataLoader(train_ds, **loader_kwargs)

    # 4. Estimator Accumulation
    # Determine rank from config and initialize estimator
    estimator = KoopmanEstimator(rank=cfg.model.n_modes, use_cca=True)

    # Prepare scaling factor (if learned singular values exist)
    scale = None
    if pl_module.svals is not None:
        scale = pl_module.svals.view(1, -1).sqrt().to(device)

    logger.info("Accumulating statistics on the full dataset...")

    with torch.no_grad():
        for batch in tqdm(inference_loader, desc="Estimator Accumulation"):
            x, y = batch

            # Move batch to device (handles dicts for molecular data)
            if isinstance(x, dict):
                x = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }
                y = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in y.items()
                }
            else:
                x, y = x.to(device), y.to(device)

            # Forward pass
            f_t = pl_module(x, lagged=False)
            g_t = pl_module(x, lagged=True)
            f_next = pl_module(y, lagged=False)
            g_next = pl_module(y, lagged=True)

            # Apply singular value scaling
            if scale is not None:
                f_t *= scale
                g_t *= scale
                f_next *= scale
                g_next *= scale

            # Extract raw data for projection/decoding
            batch_size = f_t.shape[0]
            x_raw_np = extract_raw_data(x, batch_size)
            y_raw_np = extract_raw_data(y, batch_size)

            # Accumulate covariance statistics
            estimator.partial_fit(
                f_t=f_t.cpu().numpy(),
                g_t=g_t.cpu().numpy(),
                f_next=f_next.cpu().numpy(),
                g_next=g_next.cpu().numpy(),
                x_raw=x_raw_np,
                y_raw=y_raw_np,
            )

    # 5. Finalize & Save Results
    logger.info("Finalizing Koopman Operator...")
    estimator.finalize()

    logger.info(f"Top 5 Eigenvalues: {estimator.eig()[:5]}")

    save_path = run_dir / "results.npz"
    estimator.save(save_path)
    logger.info(f"Saved operator statistics to: {save_path}")


# CLI Entry Point
if __name__ == "__main__":
    from absl import app, flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string("run_dir", None, "Path to the experiment run directory")

    def main(_):
        if not FLAGS.run_dir:
            raise ValueError("Flag --run_dir must be specified.")

        logging.basicConfig(level=logging.INFO)
        run_post_training_inference(Path(FLAGS.run_dir))

    app.run(main)
