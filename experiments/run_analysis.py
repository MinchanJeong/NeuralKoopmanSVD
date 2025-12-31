# experiments/run_analysis.py
import logging
import yaml
from pathlib import Path
from absl import app, flags
from ml_collections import ConfigDict
from experiments.pipelines import get_pipeline

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "run_dir",
    None,
    "Path to the experiment run directory (containing config.yaml and checkpoints).",
)
flags.DEFINE_string(
    "output_dir", "analysis", "Sub-directory within run_dir to save analysis results."
)


def main(_):
    if not FLAGS.run_dir:
        raise ValueError(
            "Flag --run_dir must be specified (e.g., results/project/timestamp)."
        )

    run_path = Path(FLAGS.run_dir)
    config_path = run_path / "config.yaml"

    # 1. Load Saved Config (Consistency Check)
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found in {run_path}. Cannot reproduce analysis."
        )

    logging.info(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        # Restore yaml dict to ConfigDict
        cfg_dict = yaml.unsafe_load(f)
        cfg = ConfigDict(cfg_dict)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 2. Initialize Pipeline with Restored Config
    pipeline = get_pipeline(cfg)

    # 3. Run Analysis
    analysis_out = run_path / FLAGS.output_dir
    analysis_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting analysis for run: {run_path}")
    logger.info(f"Saving results to: {analysis_out}")

    pipeline.analyze(run_dir=run_path, output_dir=analysis_out)


if __name__ == "__main__":
    app.run(main)
