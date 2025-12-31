# experiments/pipelines/__init__.py
from .chignolin import ChignolinPipeline
from .synthetic import SyntheticPipeline
from .mnist import MnistPipeline


def get_pipeline(cfg):
    data_type = cfg.data.type

    if data_type == "molecular":
        return ChignolinPipeline(cfg)
    elif data_type == "synthetic":
        return SyntheticPipeline(cfg)
    elif cfg.data.type == "mnist":
        return MnistPipeline(cfg)
    else:
        raise ValueError(f"No pipeline found for data type: {data_type}")
