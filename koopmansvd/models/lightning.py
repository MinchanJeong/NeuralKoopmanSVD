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
        """
        Executes a single training step.

        In a DDP (Distributed Data Parallel) setting, we perform an `all_gather` operation
        to collect features from all GPUs before calculating the loss.
        This is crucial because objectives like LoRA, VAMP, and DPNet
        rely on second-moment matrices (e.g., E_{x,y' ~ D_train}[f(x) g(x')^T] ),
        which are quadratic statistics that require global batch information.

        Calculating the loss on local mini-batches and averaging the results (Avg(Loss(M_local)))
        is mathematically different from calculating the loss on the global batch (Loss(Avg(M_local))).
        The latter (Global Batch) provides a rigorous and stable gradient estimate for
        learning singular subspaces.
        """

        # Batch is expected to be (x, y) tuple from KoopmanCollate
        x, y = batch

        # Forward Pass
        f_x = self(x, lagged=False)
        g_y = self(y, lagged=True)
        batch_size = f_x.shape[0]

        if self.trainer.world_size > 1:
            """
            < Efficient Differentiable All-Gather >

            - Problem with `nn.functional.all_gather`:
            It maintains the computational graph for ALL gathered tensors. During backward(),
            it triggers an all-to-all communication to synchronize gradients for remote samples.
            However, for covariance-based losses (VAMP, LoRA), the gradient of the loss
            with respect to local samples depends only on the local batch.

            - Solution (The SimCLR Trick):
            1. Gather raw data without gradients (stop-gradient).
            2. Manually overwrite the current rank's slice with the local differentiable tensor.
            This forces the autograd engine to backpropagate ONLY through the local batch,
            eliminating massive redundant gradient communication.
            """

            # 1. Create placeholders
            f_gathered = [torch.zeros_like(f_x) for _ in range(self.trainer.world_size)]
            g_gathered = [torch.zeros_like(g_y) for _ in range(self.trainer.world_size)]

            # 2. Collect values (No gradient flow here)
            dist.all_gather(f_gathered, f_x)
            dist.all_gather(g_gathered, g_y)

            # 3. Inject local differentiable tensors into the gathered list
            # This attaches the current GPU's computational graph to the global batch container.
            f_gathered[self.trainer.global_rank] = f_x
            g_gathered[self.trainer.global_rank] = g_y

            # 4. Form the global batch
            f_x_global = torch.cat(f_gathered, dim=0)
            g_y_global = torch.cat(g_gathered, dim=0)

            # Calculate loss on the global batch
            loss_val = self.loss_fn(f_x_global, g_y_global, svals=self.svals)
            # Scale loss by world size to account for gradient accumulation
            loss = loss_val * self.trainer.world_size
            batch_size *= self.trainer.world_size
        else:
            # Loss Calculation
            # Pass svals if available (Loss fn decides how to use it)
            loss_val = self.loss_fn(f_x, g_y, svals=self.svals)
            loss = loss_val

        self.log(
            "train/loss",
            loss_val,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        # Update Training Metric (Naive VAMP-2)
        # Detach svals to avoid double backprop through metric accumulation
        if self.svals is not None:
            scale = self.svals.view(1, -1).sqrt().detach()
            self.train_metric.update(f_x.detach() * scale, g_y.detach() * scale)
        else:
            self.train_metric.update(f_x.detach(), g_y.detach())

        # Log VAMP-2 Score for Training Convergence Monitoring
        # Note: on_step=False ensures it's computed once per epoch to save SVD cost
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
            self.log_dict(
                log_scores, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
            )
        else:
            # Fallback
            self.log(
                "val/vamp_2_naive",
                scores,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.val_metric.reset()

    def configure_optimizers(self):
        return self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
