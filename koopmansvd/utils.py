# koopmansvd/utils.py
import os
import sys
import logging
import torch


# --- Logging Utilities ---
class RankZeroFilter(logging.Filter):
    """Filters out logs from non-master processes during runtime."""

    def filter(self, record):
        # Check standard DDP initialization
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return False
        if "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != 0:
            return False
        return True


def setup_logger(run_dir):
    """Configures root logger with file (Rank 0 only) and console (Filtered) handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File Handler: Only on Rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        log_file = run_dir / "train_log.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    # Console Handler: Filtered
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.addFilter(RankZeroFilter())
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def resolve_device_count(cfg):
    """
    Determines the number of devices (GPUs) to be used based on config.
    Returns: (int) num_devices
    """
    req_devices = cfg.trainer.devices
    accelerator = cfg.trainer.accelerator

    if req_devices == "auto" or req_devices == -1:
        if accelerator == "gpu" and torch.cuda.is_available():
            return torch.cuda.device_count()
        return 1
    elif isinstance(req_devices, list):
        return len(req_devices)
    else:
        return int(req_devices)
