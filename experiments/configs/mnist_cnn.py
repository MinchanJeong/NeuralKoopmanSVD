import os
from experiments.configs import default


def get_config():
    config = default.get_config()

    # --- Data ---
    config.data.type = "mnist"
    config.data.dataset_name = "mnist"

    data_root = os.environ.get("DATA_DIR", "./data")
    config.data.path = os.path.join(data_root, "ordered_mnist.npy")

    config.data.global_batch_size = 64
    config.data.split = (0.8, 0.2)

    # --- Model ---
    config.model.type = "cnn"
    config.model.n_modes = 10
    config.model.centering = True

    # --- Encoder ---
    config.model.encoder.cnn.input_channels = 1
    config.model.encoder.cnn.hidden_channels = (16, 32)

    # --- Loss ---
    config.model.loss.type = "lora"
    config.model.loss.nesting = "jnt"

    # --- Optimization ---
    config.optimization.lr = 1e-3
    config.optimization.learn_svals = False

    # --- Trainer ---
    config.trainer.max_epochs = 100

    return config
