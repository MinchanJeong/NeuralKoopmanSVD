# experiments/run_preprocessing.py
import logging
from absl import app, flags
from ml_collections import config_flags
from experiments.pipelines import get_pipeline

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "experiments/configs/default.py",
    "Path to the config file.",
    lock_config=True,
)
flags.DEFINE_boolean("overwrite", False, "Overwrite existing processed data.")


def main(_):
    cfg = FLAGS.config
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info(f"Initializing Pipeline for: {cfg.data.type}")

    # Initialize pipeline via factory
    pipeline = get_pipeline(cfg)

    # Execute preprocessing
    pipeline.preprocess(overwrite=FLAGS.overwrite)


if __name__ == "__main__":
    app.run(main)
