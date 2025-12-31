# experiments/configs/synthetic_logistic.py
from experiments.configs import default


def get_config():
    config = default.get_config()

    # Data Settings
    config.data.type = "synthetic"
    config.data.dataset_name = "logistic"
    config.data.path = "./data/logistic_traj.npy"
    config.data.n_samples = 50000

    config.data.global_batch_size = 256

    # Model Settings (MLP)
    config.model.type = "mlp"
    config.model.n_modes = 20

    # Define a pre-encoder embedding layer specifically for this experiment.
    # The factory will compose this with the main MLP encoder.
    config.model.encoder.embedding_type = "sinusoidal"

    config.model.encoder.mlp.hidden_dims = (64, 128, 64)
    config.model.encoder.mlp.activation = "LeakyReLU"
    config.model.encoder.mlp.use_batchnorm = False

    # Optimization
    config.optimization.lr = 1e-3
    config.model.loss.type = "lora"
    config.model.loss.nesting = "jnt"

    config.trainer.max_epochs = 200

    return config
