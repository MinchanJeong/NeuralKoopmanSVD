# koopmansvd/experiments/train.py

import logging
import datetime
from pathlib import Path
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from absl import app, flags
from ml_collections import config_flags

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from experiments.factory import (
    get_datasets,
    get_encoder,
    get_loss,
    get_lightning_module,
)
from koopmansvd.models.inference import KoopmanEstimator
from koopmansvd.utils import setup_logger, resolve_device_count
from koopmansvd.data.utils import extract_raw_data

# --- Configuration ---
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "experiments/configs/default.py",
    "Path to the config file.",
    lock_config=True,
)
flags.DEFINE_string("workdir", "./results", "Directory to store results.")
flags.DEFINE_string(
    "run_id",
    None,
    "Unique run identifier (e.g., timestamp). Required for DDP consistency.",
)


def resolve_loader_kwargs(cfg):
    """
    Dynamically resolves DataLoader parameters to balance throughput and memory.
    """
    # 1. Resolve num_workers
    # If set to -1, auto-scale to CPU count (capped at 8-16 to avoid overhead)
    if cfg.data.num_workers == -1:
        cpu_count = os.cpu_count() or 1
        # Heuristic: usually 4 workers per GPU is sufficient.
        # Capping at 8 or 12 prevents excessive context switching overhead.
        cfg.data.num_workers = min(cpu_count, 12)

    num_workers = cfg.data.num_workers

    # 2. Resolve prefetch_factor
    # prefetch_factor is valid ONLY if num_workers > 0
    if num_workers > 0:
        # Use config value, defaulting to 2 if not present
        prefetch_factor = getattr(cfg.data, "prefetch_factor", 2)
        persistent_workers = getattr(cfg.data, "persistent_workers", True)
    else:
        # Main process loading (debugging)
        prefetch_factor = None
        persistent_workers = False

    logging.info(
        f"DataLoader Config: Workers={num_workers}, Prefetch={prefetch_factor}, Persistent={persistent_workers}"
    )

    return {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": persistent_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }


# --- Main Training Execution ---
def main(_):
    cfg = FLAGS.config

    # 1. Environment Setup
    # DDP-safe run_id generation and directory creation.
    is_ddp = dist.is_available() and dist.is_initialized()
    run_id = None

    # Rank 0 generates the ID, which is then broadcast to all other processes.
    if not is_ddp or dist.get_rank() == 0:
        if FLAGS.run_id:
            run_id = FLAGS.run_id
        else:
            # Generate a descriptive run_id using a tag from the config
            tag = cfg.data.dataset_name if cfg.data.dataset_name else cfg.data.type
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"{tag}_{timestamp}"

    if is_ddp:
        # Wrap run_id in a list for broadcast
        run_id_list = [run_id]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]

    # At this point, all processes have the same run_id.
    # Create directory only on Rank 0.
    run_dir = Path(FLAGS.workdir) / cfg.project_name / run_id
    if not is_ddp or dist.get_rank() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)

    # All processes wait until the directory is created.
    if is_ddp:
        dist.barrier()

    logger = setup_logger(run_dir)
    logger.info(f"Experiment Directory: {run_dir}")

    L.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    num_devices = resolve_device_count(cfg)
    global_batch_size = cfg.data.global_batch_size
    if global_batch_size % num_devices != 0:
        logging.warning(
            f"Global Batch Size ({global_batch_size}) not divisible by devices ({num_devices}). "
            f"Local BS will be {global_batch_size // num_devices}."
        )
    cfg.data.batch_size = global_batch_size // num_devices

    logging.info("Batch Size Configuration:")
    logging.info(f"- Global Batch Size: {global_batch_size}")
    logging.info(f"- Num Devices: {num_devices}")
    logging.info(f"- Per-Device Batch Size: {cfg.data.batch_size}")

    # 2. Data Pipeline
    train_ds, val_ds, collate_fn = get_datasets(cfg)

    loader_kwargs = resolve_loader_kwargs(cfg)
    loader_kwargs["batch_size"] = cfg.data.batch_size
    loader_kwargs["collate_fn"] = collate_fn

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    sample_x = train_ds[0][0]
    input_dim = sample_x.numel() if isinstance(sample_x, torch.Tensor) else None
    cfg.data.input_dim = input_dim

    # 3. Model Setup
    # Dynamically set input_dim if the corresponding encoder config has the key.
    model_type = cfg.model.type
    if model_type in cfg.model.encoder and "input_dim" in cfg.model.encoder[model_type]:
        cfg.model.encoder[model_type].input_dim = input_dim

    # Create directory and save config (Rank 0 only)
    if not is_ddp or dist.get_rank() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            f.write(cfg.to_yaml())

    logger.info("Building Model...")
    encoder = get_encoder(cfg)
    lagged_encoder = get_encoder(cfg)
    loss_fn = get_loss(cfg)

    pl_module = get_lightning_module(cfg, encoder, lagged_encoder, loss_fn)

    lightning_loggers = []
    csv_logger = CSVLogger(save_dir=str(run_dir), name="logs", version="")
    lightning_loggers.append(csv_logger)

    if cfg.logging.use_wandb:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "Config sets 'use_wandb=True' but 'wandb' is not installed.\n"
                "Please install it via: pip install wandb\n"
                "Or set 'logging.use_wandb=False' in your config."
            )
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=FLAGS.run_id,  # Use the generated run_id
            config=cfg.to_dict(),
            save_dir=str(run_dir),
        )
        if not is_ddp or dist.get_rank() == 0:
            additional_config = {
                "hardware/num_devices": num_devices,
                "hardware/global_batch_size": global_batch_size,
                "hardware/local_batch_size": cfg.data.batch_size,
            }
            wandb_logger.log_hyperparams(additional_config)
        lightning_loggers.append(wandb_logger)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch-{epoch:03d}-step-{step}",
        save_top_k=1,
        monitor="val/vamp_e_score",
        mode="max",
        save_last=True,
    )

    # 4. Training
    logger.info("Starting Training...")
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        default_root_dir=str(run_dir),
        strategy="ddp" if num_devices > 1 else "auto",
        callbacks=[checkpoint_callback],
        logger=lightning_loggers,
        log_every_n_steps=cfg.logging.log_every_n_steps,
    )
    trainer.fit(pl_module, train_loader, val_loader)

    # 5. Post-Training Inference
    # Calculate operator statistics on the full training set for stable inference.
    # Only Rank 0 performs this to avoid redundant computation and file writes.
    if trainer.is_global_zero:
        logger.info("Running Post-Training Inference to compute Operator Statistics...")

        pl_module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pl_module.to(device)

        # Determine effective rank from config
        estimator = KoopmanEstimator(rank=cfg.model.n_modes, use_cca=True)

        inference_loader_kwargs = loader_kwargs.copy()
        inference_loader_kwargs["shuffle"] = False
        inference_loader_kwargs["drop_last"] = False

        inference_loader = DataLoader(train_ds, **inference_loader_kwargs)

        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Estimator Accumulation"):
                x, y = batch  # x: state(t), y: state(t+1)

                # Move batch to device
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

                # Extract features for current (t) and next (t+1) steps
                # Both encoder (f) and lagged encoder (g) are needed for EDMD/CCA cross-validation
                f_t = pl_module(x, lagged=False)
                g_t = pl_module(x, lagged=True)
                f_next = pl_module(y, lagged=False)
                g_next = pl_module(y, lagged=True)

                # Apply learned singular value scaling if applicable
                if pl_module.svals is not None:
                    scale = pl_module.svals.view(1, -1).sqrt()
                    f_t *= scale
                    g_t *= scale
                    f_next *= scale
                    g_next *= scale

                # Extract raw data for projection (decoding)
                batch_size = f_t.shape[0]
                x_raw_np = extract_raw_data(x, batch_size)
                y_raw_np = extract_raw_data(y, batch_size)

                # Accumulate statistics
                estimator.partial_fit(
                    f_t=f_t.cpu().numpy(),
                    g_t=g_t.cpu().numpy(),
                    f_next=f_next.cpu().numpy(),
                    g_next=g_next.cpu().numpy(),
                    x_raw=x_raw_np,
                    y_raw=y_raw_np,
                )

        logger.info("Finalizing Koopman Operator...")
        estimator.finalize()

        # Log top eigenvalues for sanity check
        logger.info(f"Top 5 Eigenvalues: {estimator.eig()[:5]}")

        save_path = run_dir / "results.npz"
        estimator.save(save_path)
        logger.info(f"Results saved to {save_path}")

    if is_ddp:
        dist.barrier()

    if cfg.logging.use_wandb and trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
