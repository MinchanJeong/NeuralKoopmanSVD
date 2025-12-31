import os
import ml_collections


def get_config():
    """
    Returns the default configuration using nested dictionary structure.
    Acts as the schema and base configuration for all experiments.
    """
    return ml_collections.ConfigDict(
        {
            # --- Global Settings ---
            "seed": 42,
            "project_name": "KoopmanSVD",
            # Default to ./results if RESULT_DIR env var is not set
            "workdir": os.environ.get("RESULT_DIR", "./results"),
            "logging": {
                "use_wandb": False,  # Set to True via CLI to enable
                "wandb_project": "KoopmanSVD",
                "wandb_entity": None,  # Optional: Team/User entity
                "log_every_n_steps": 50,  # Log frequency
            },
            # --- Data Configuration ---
            "data": {
                # > General
                "seed": 0,
                "type": "tensor",  # 'tensor', 'molecular', 'synthetic', 'mnist'
                "format": "auto",  # 'auto', 'npy', 'zarr'
                "path": "",
                "dataset_name": None,
                # > Loading & Splitting
                "global_batch_size": 128,
                "batch_size": 128,  # Per-device batch size (updated in train.py)
                "num_workers": 4,
                "time_lag": 1,
                "split": (0.8, 0.2),
                # > Synthetic Specific
                "n_samples": 50000,
                "input_dim": None,
            },
            # --- Model Configuration ---
            "model": {
                "type": "mlp",  #
                "n_modes": 16,  # Rank (Number of singular values)
                "centering": True,  # Apply CenteringWrapper?
                # Encoder Hyperparameters (Union of MLP and SchNet params)
                "encoder": {
                    "embedding_type": None,
                    "mlp": {
                        "input_dim": None,  # If input_dim is defined, it is set dynamically in train.py
                        "hidden_dims": (128, 128),
                        "activation": "ReLU",
                        "use_batchnorm": True,
                    },
                    "cnn": {
                        "input_channels": 1,
                        "hidden_channels": (16, 32, 64),
                        "use_batchnorm": False,
                    },
                    "schnet": {
                        "n_atom_basis": 64,
                        "n_interactions": 3,
                        "n_rbf": 20,
                        "cutoff": 6.0,
                        "aggregation": "mean",
                        "use_batchnorm": True,
                    },
                },
                # Loss Function Configuration
                "loss": {
                    "type": "lora",  # 'lora', 'vamp', 'dp'
                    # LoRA params
                    "nesting": "jnt",  # 'jnt', 'seq', None
                    # DPLoss / VAMP params
                    "relaxed": True,
                    "metric_deformation": 0.01,
                    "reg_weight": 0.0,
                    "schatten_norm": 2,
                    "center_covariances": False,
                },
            },
            # --- Optimization Configuration ---
            "optimization": {
                "lr": 1e-3,
                "learn_svals": False,  # For LoRA: Learn singular values?
                "svals_init_mode": "sigmoid",  # 'sigmoid', 'exp', 'softplus'
            },
            # --- Trainer (PyTorch Lightning) Configuration ---
            "trainer": {
                "max_epochs": 100,
                "accelerator": "gpu",
                "devices": 1,  # should be 1 if we use torchrun on execution
                "check_val_every_n_epoch": 5,
                "gradient_clip_val": 0.0,
            },
        }
    )
