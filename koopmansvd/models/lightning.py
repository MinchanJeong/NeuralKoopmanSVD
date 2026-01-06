# koopmansvd/models/lightning.py
import logging
from typing import Optional, Dict, Any, Type
import torch
import torch.distributed as dist
import torch.nn as nn
import lightning as L
from torch.optim import Optimizer

from koopmansvd.models.encoders.base import BaseEncoder
from koopmansvd.models.metrics import KoopmanScoreMetric
from koopmansvd.models.inference import linalg

logger = logging.getLogger(__name__)


class KoopmanPLModule(L.LightningModule):
    """
    Unified LightningModule for training Koopman Operators.
    Supports:
      - Generic Encoders (MLP, SchNet, etc.)
      - Lagged Encoders (Shared or Separate)
      - Learned Singular Values (LoRA)
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        loss_fn: nn.Module,
        optimizer_cls: Type[Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None,
        lagged_encoder: Optional[BaseEncoder] = None,
        learn_svals: bool = False,
        num_svals: Optional[int] = None,  # If None, inferred from encoder
        svals_init_mode: str = "sigmoid",  # 'sigmoid', 'exp', 'softplus'
        has_centering: bool = False,  # Explicit flag
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "lagged_encoder", "loss_fn"])

        # 1. Encoders
        self.encoder = encoder
        self.lagged_encoder = lagged_encoder if lagged_encoder else encoder

        # 2. Loss & Optimization
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}

        # 3. Learned Singular Values (LoRA specific)
        self.learn_svals = learn_svals
        self.svals_init_mode = svals_init_mode
        self.has_centering = has_centering

        self.train_metric = KoopmanScoreMetric(feature_dim=self.encoder.output_dim)
        self.val_metric = KoopmanScoreMetric(feature_dim=self.encoder.output_dim)

        if learn_svals:
            # Total dimension of features
            total_dim = num_svals if num_svals is not None else self.encoder.output_dim
            # If centering is enabled, we fix the first singular value to 1.0
            # and learn the remaining (total_dim - 1) values.
            self.num_learnable_svals = (
                total_dim - 1 if self.has_centering else total_dim
            )

            init_vals = torch.sort(
                torch.randn(self.num_learnable_svals), descending=True
            ).values
            self.svals_params = nn.Parameter(init_vals)
        else:
            self.register_parameter("svals_params", None)

    @property
    def svals(self) -> Optional[torch.Tensor]:
        """Returns the actual singular values (processed)."""
        if not self.learn_svals:
            return None
        # 1. Apply activation to enforce positivity
        if self.svals_init_mode == "sigmoid":
            vals = torch.sigmoid(self.svals_params)
        elif self.svals_init_mode == "exp":
            vals = torch.exp(self.svals_params)
        elif self.svals_init_mode == "softplus":
            vals = torch.nn.functional.softplus(self.svals_params)
        else:
            vals = self.svals_params  # Raw

        # Prepend 1.0 explicitly if centering is enabled
        if self.has_centering:
            ones = torch.ones(1, device=vals.device, dtype=vals.dtype)
            vals = torch.cat([ones, vals])

        return vals

    def forward(self, x, lagged: bool = False):
        enc = self.lagged_encoder if lagged else self.encoder
        return enc(x)

    def training_step(self, batch, batch_idx):
        # Batch is expected to be (x, y) tuple from KoopmanCollate
        x, y = batch

        # Forward Pass
        f_x = self(x, lagged=False)
        g_y = self(y, lagged=True)

        # Loss Calculation
        # Pass svals if available (Loss fn decides how to use it)
        loss = self.loss_fn(f_x, g_y, svals=self.svals)

        # Update Training Metric with f, g
        # Detach to avoid double backprop through metric
        if self.svals is not None:
            scale = self.svals.view(1, -1).sqrt().detach()
            self.train_metric.update(f_x.detach() * scale, g_y.detach() * scale)
        else:
            self.train_metric.update(f_x.detach(), g_y.detach())

        batch_size = f_x.shape[0]
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Log VAMP-2 Score for Training Convergence Monitoring
        self.log(
            "train/vamp_2",
            self.train_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if self.learn_svals:
            self.log(
                "train/sval_max",
                self.svals.max(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/sval_min",
                self.svals.min(),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        return loss

    def on_train_epoch_start(self):
        self.train_metric.reset()
        self.val_metric.set_ref_stats(None)

    def on_validation_epoch_start(self):
        """
        Aggregates global training statistics via All-Reduce to compute the
        reference Koopman operator for VAMP-E validation.
        """
        # 1. Clone local stats to avoid in-place modification of the metric state
        n_samples = self.train_metric.n_samples.clone()
        M_f = self.train_metric.accum_M_f.clone()
        M_g = self.train_metric.accum_M_g.clone()
        T_fg = self.train_metric.accum_T_fg.clone()

        # 2. Synchronize: Sum statistics across all DDP processes
        if self.trainer.world_size > 1:
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(M_f, op=dist.ReduceOp.SUM)
            dist.all_reduce(M_g, op=dist.ReduceOp.SUM)
            dist.all_reduce(T_fg, op=dist.ReduceOp.SUM)

        # Sanity check: Skip if insufficient data
        if n_samples < 2:
            return

        # 3. Compute global means and move to CPU for linalg operations
        # (Since dist.all_reduce ensures sync, all ranks get identical values here)
        M_f_np = (M_f / n_samples).cpu().numpy()
        M_g_np = (M_g / n_samples).cpu().numpy()
        T_fg_np = (T_fg / n_samples).cpu().numpy()

        stats = linalg.OperatorStats(M_f_np, M_g_np, T_fg_np)

        # 4. Perform CCA and inject reference stats into val_metric
        try:
            cca_components = linalg.perform_cca(stats)
            self.val_metric.set_ref_stats(cca_components)
        except Exception as e:
            if self.trainer.is_global_zero:
                print(f"Warning: CCA computation for VAMP-E failed: {e}")
            self.val_metric.set_ref_stats(None)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        f_x = self(x, lagged=False)
        g_y = self(y, lagged=True)

        # Compute Validation Metric. VAMP-E score will be computed.
        # Even if training with LoRA, we might want to monitor VAMP
        # Apply svals for metric calculation if they exist
        if self.svals is not None:
            scale = self.svals.sqrt()
            f_x = f_x * scale
            g_y = g_y * scale

        # Update Metric with f, g
        self.val_metric.update(f_x, g_y)

    def on_validation_epoch_end(self):
        """
        Manually handle validation metric logging because compute() returns a Dict.
        """
        # compute() returns {'vamp_e': ..., 'vamp_2': ...}
        scores = self.val_metric.compute()

        # val/vamp_e, val/vamp_2
        if isinstance(scores, dict):
            log_scores = {f"val/{k}": v for k, v in scores.items()}
            self.log_dict(log_scores, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Fallback
            self.log(
                "val/vamp_2_naive", scores, on_step=False, on_epoch=True, prog_bar=True
            )

        self.val_metric.reset()

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
