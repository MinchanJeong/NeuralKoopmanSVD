# experiments/factory.py
import logging
import inspect
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import random_split

from koopmansvd.data import TensorContextDataset, MolecularContextDataset
from koopmansvd.data.io import load_trajectory_data
from koopmansvd.data.collate import KoopmanCollate, default_tensor_collate
from koopmansvd.data.molecular import schnet_collate_fn

from koopmansvd.models.encoders import (
    MLPEncoder,
    CNNEncoder,
    SchNetSystemEncoder,
    CenteringWrapper,
    BatchL2NormalizationWrapper,
    SinusoidalEmbedding,
)
from koopmansvd.models import (
    NestedLoRALoss,
    VAMPLoss,
    DPLoss,
    KoopmanPLModule,
)

logger = logging.getLogger(__name__)


def _instantiate_component(cls, config_dict, **extra_args):
    """
    Dynamically instantiates a class by filtering 'config_dict' to match
    the class's __init__ signature.

    This avoids manual 'pop' operations and makes the factory scalable
    to new models with different hyperparameters.
    """
    # 1. Get the signature of the constructor
    sig = inspect.signature(cls.__init__)

    # update config_dict with extra_args
    config_dict_copy = config_dict.copy()
    config_dict_copy.update(extra_args)

    # 2. Filter config_dict to keep only arguments present in the signature
    # Note: We skip 'self' automatically.
    # If the class has **kwargs, we could technically pass everything,
    # but strict filtering is safer to prevent silent errors.
    valid_args = {k: v for k, v in config_dict_copy.items() if k in sig.parameters}

    ignored = set(config_dict_copy.keys()) - set(valid_args.keys())
    if ignored:
        logger.warning(f"Ignored config keys for {cls.__name__}: {ignored}")

    # 4. Check for missing required arguments (optional safety)
    # This prevents obscure errors later if a required arg is missing from config
    for param_name, param in sig.parameters.items():
        if (
            param_name != "self"
            and param.default == inspect.Parameter.empty
            and param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param_name not in valid_args
        ):
            raise ValueError(
                f"Missing required argument '{param_name}' for {cls.__name__}"
            )

    return cls(**valid_args)


def get_datasets(cfg):
    """
    Factory for Datasets and Collate Function.
    Returns:
        train_ds, val_ds, collate_fn
    """
    if cfg.data.type == "molecular":
        if not cfg.data.train_db_path or not cfg.data.val_db_path:
            split_name = f"split_{int(cfg.data.split[0] * 100)}_seed{cfg.data.seed}"
            db_dir = Path(cfg.data.dataset_dir) / split_name
            cfg.data.train_db_path = str(db_dir / f"train_{cfg.data.mode}.db")
            cfg.data.val_db_path = str(db_dir / f"val_{cfg.data.mode}.db")

        logger.info("Loading Molecular Data...")
        logger.info(f"  Train: {cfg.data.train_db_path}")
        logger.info(f"  Val:   {cfg.data.val_db_path}")

        train_ds = _instantiate_component(
            MolecularContextDataset,
            cfg.data.to_dict(),
            db_path=cfg.data.train_db_path,
            cutoff=cfg.model.encoder.schnet.cutoff,
        )
        val_ds = _instantiate_component(
            MolecularContextDataset,
            cfg.data.to_dict(),
            db_path=cfg.data.val_db_path,
            cutoff=cfg.model.encoder.schnet.cutoff,
        )

        # Inject SchNet-specific collation logic
        collate_fn = KoopmanCollate(base_collate_fn=schnet_collate_fn)

        return train_ds, val_ds, collate_fn

    else:
        logger.info(f"Loading Tensor Data: {cfg.data.path}")
        if not Path(cfg.data.path).exists():
            raise FileNotFoundError(f"Data file not found: {cfg.data.path}")

        data = load_trajectory_data(cfg)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        full_ds = _instantiate_component(
            TensorContextDataset, cfg.data.to_dict(), data=data
        )

        train_len = int(len(full_ds) * cfg.data.split[0])
        val_len = len(full_ds) - train_len

        train_ds, val_ds = random_split(
            full_ds,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(cfg.data.seed),
        )

        collate_fn = KoopmanCollate(base_collate_fn=default_tensor_collate)

        return train_ds, val_ds, collate_fn


def get_encoder(cfg, **kwargs):
    """Factory for Encoders using Introspection and Composition."""
    effective_n_modes = cfg.model.n_modes
    if cfg.model.centering:
        effective_n_modes -= 1

    model_type = cfg.model.type
    if model_type not in cfg.model.encoder:
        raise ValueError(f"No encoder config found for type: {model_type}")

    enc_config = cfg.model.encoder[model_type].to_dict()

    encoder_layers = []
    current_input_dim = cfg.data.input_dim

    # 1. Build Embedding Layer if specified
    embedding_type = cfg.model.encoder.get("embedding_type")
    if embedding_type == "sinusoidal":
        if current_input_dim is None:
            raise ValueError("cfg.data.input_dim must be set for SinusoidalEmbedding.")
        embedding = SinusoidalEmbedding(input_dim=current_input_dim)
        # The main encoder's input is now the embedding's output
        current_input_dim = embedding.output_dim
    else:
        embedding = None

    # 2. Build Main Encoder
    if model_type == "schnet":
        main_encoder = _instantiate_component(
            SchNetSystemEncoder, enc_config, output_dim=effective_n_modes
        )
    elif model_type == "cnn":
        main_encoder = _instantiate_component(
            CNNEncoder, enc_config, output_dim=effective_n_modes
        )
    elif model_type == "mlp":
        main_encoder = _instantiate_component(
            MLPEncoder, enc_config, output_dim=effective_n_modes
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    encoder_layers.append(main_encoder)

    # 3. Compose Encoder
    if len(encoder_layers) > 1:
        encoder = nn.Sequential(*encoder_layers)
        # Attach output_dim property for compatibility with downstream components
        encoder.output_dim = main_encoder.output_dim
    else:
        encoder = main_encoder

    # 4. Apply Wrappers
    if cfg.model.loss.type == "lora" and cfg.optimization.learn_svals:
        encoder = BatchL2NormalizationWrapper(encoder)

    if cfg.model.centering:
        encoder = CenteringWrapper(encoder)

    return encoder


def get_loss(cfg):
    loss_config = cfg.model.loss.to_dict()

    if cfg.model.loss.type == "lora":
        return _instantiate_component(
            NestedLoRALoss, loss_config, n_modes=cfg.model.n_modes
        )
    elif cfg.model.loss.type == "vamp":
        return _instantiate_component(VAMPLoss, loss_config)
    elif cfg.model.loss.type == "dp":
        return _instantiate_component(DPLoss, loss_config)
    else:
        raise ValueError(f"Unknown loss type: {cfg.model.loss.type}")


def get_lightning_module(cfg, encoder, lagged_encoder, loss_fn):
    """Factory to select between Standard and Langevin PL Modules"""

    base_kwargs = {
        "encoder": encoder,
        "lagged_encoder": lagged_encoder,
        "loss_fn": loss_fn,
        "optimizer_kwargs": {"lr": cfg.optimization.lr},
        "learn_svals": cfg.optimization.learn_svals,
        "num_svals": cfg.model.n_modes,
        "has_centering": cfg.model.centering,
    }

    return KoopmanPLModule(**base_kwargs)
