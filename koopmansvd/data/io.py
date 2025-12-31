# koopmansvd/data/io.py
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class DataLoaders:
    """
    Registry for data loaders.
    Returns a 'sliceable' object (Array-like) that supports lazy access.
    """

    @staticmethod
    def load_numpy(path: str, lazy: bool = True):
        """Loads .npy files. Uses mmap if lazy=True."""
        if lazy:
            try:
                data = np.load(path, mmap_mode="r")
                logger.info(f"Loaded {path} with mmap_mode='r'. Shape: {data.shape}")
                return data
            except ValueError:
                logger.warning(f"Could not mmap {path}. Falling back to full load.")

        return np.load(path)

    @staticmethod
    def load_zarr(path: str, lazy: bool = True):
        """Loads Zarr/NetCDF data via Xarray. Ideal for ERA5."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "Loading Zarr/NetCDF requires 'xarray' and 'zarr'. Pip install them."
            )

        # Climate data files can be large; lazy loading is preferred.
        # chunks='auto' lets xarray/dask decide chunk sizes.
        ds = xr.open_dataarray(
            path, chunks="auto", engine="zarr" if path.endswith(".zarr") else None
        )

        logger.info(f"Loaded Zarr/NetCDF from {path}. Shape: {ds.shape}")

        # Note: Xarray DataArray supports slicing directly.
        # If Dataset (multiple variables), user must select variable first.
        return ds

    @staticmethod
    def load_auto(path: str, lazy: bool = True):
        """Dispatches based on file extension."""
        ext = os.path.splitext(path)[-1].lower()

        if ext == ".npy":
            return DataLoaders.load_numpy(path, lazy)
        elif ext in [".zarr", ".nc", ".nc4"]:
            return DataLoaders.load_zarr(path, lazy)
        else:
            # Fallback for generic binary or unknown types -> Try numpy
            logger.warning(f"Unknown extension {ext}. Trying NumPy loader.")
            return DataLoaders.load_numpy(path, lazy)


def load_trajectory_data(cfg):
    """
    Unified entry point for data loading.

    Args:
        cfg: ConfigDict containing cfg.data.path and optionally cfg.data.format
    """
    path = cfg.data.path
    fmt = getattr(cfg.data, "format", "auto")

    if fmt == "numpy":
        return DataLoaders.load_numpy(path)
    elif fmt == "zarr":
        return DataLoaders.load_zarr(path)
    else:  # auto
        return DataLoaders.load_auto(path)
