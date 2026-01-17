import os
from experiments.configs import default


def get_config():
    """
    Configuration for Chignolin protein dynamics using SchNet + LoRA.
    """
    # 1. Load Base Config
    config = default.get_config()

    # Data Settings
    config.data.type = "molecular"

    # Path Handling: Prioritize Env Vars -> Relative Defaults
    data_root = os.environ.get("DATA_DIR", "./data")
    config.data.raw_path = os.environ.get(
        "RAW_DIR", os.path.join(data_root, "chignolin_raw")
    )
    config.data.dataset_dir = os.environ.get(
        "PROCESSED_DIR", os.path.join(data_root, "chignolin_processed")
    )

    config.data.mode = "full"
    config.data.train_db_path = ""  # Constructed dynamically in pipeline
    config.data.val_db_path = ""  # Constructed dynamically in pipeline
    config.data.num_workers = 8
    config.data.prefetch_factor = 4

    config.data.global_batch_size = 384
    config.data.time_lag = 1

    # Options: "random", "distribution_matching"
    # "random" performs a simple random split controlled by the data seed.
    # "distribution_matching" performs 100 candidate splits and selects the one
    # that minimizes the Wasserstein distance between the RMSD distributions
    # of the train and validation sets.
    config.data.split_strategy = "distribution_matching"

    # Model Settings
    config.model.type = "schnet"
    config.model.n_modes = 16
    config.model.encoder.schnet.n_atom_basis = 64
    config.model.encoder.schnet.n_interactions = 3
    config.model.encoder.schnet.aggregation = "mean"

    # Optimization & Loss
    config.optimization.lr = 1e-3
    config.model.loss.type = "lora"
    config.model.loss.nesting = "jnt"
    config.model.loss.reg_weight = 0.0
    config.model.loss.metric_deformation = 0.01  # DPLoss regularization strength

    # Trainer Settings
    config.trainer.max_epochs = 200

    return config
