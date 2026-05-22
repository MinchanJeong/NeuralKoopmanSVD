# koopmansvd/models/lightning.py
import logging
from typing import Optional, Dict, Any, Type, Callable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.optim import Optimizer

import lightning as L
from lightning.pytorch.utilities import rank_zero_info

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

        # Placeholder for Ref Loader
        self._ref_dataset = None
        self._ref_loader_kwargs = {}
        self._ref_loader = None  # Created in setup()

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

    def _assemble_global_batch(self, f_x, g_y, world_size, global_rank):
        """Scale local features by sqrt(svals), then all-gather into the global batch.

        The svals scaling is applied to the LOCAL (differentiable) slice *before* the
        all-gather, and only that slice stays differentiable (the SimCLR trick). This
        keeps the svals gradient flowing through the local batch only -- consistent with
        the encoder gradient under DDP's mean-reduce plus the world_size loss scaling.
        If the scaling instead happened inside loss_fn on the gathered batch, every rank
        would compute the full global-batch svals gradient and it would be inflated by
        world_size. The caller passes svals=None to loss_fn. Returns (f_global, g_global).
        """
        svals = self.svals
        if svals is not None:
            s = svals.view(1, -1).sqrt()
            f_local, g_local = f_x * s, g_y * s
        else:
            f_local, g_local = f_x, g_y

        f_gathered = [torch.zeros_like(f_local) for _ in range(world_size)]
        g_gathered = [torch.zeros_like(g_local) for _ in range(world_size)]
        dist.all_gather(f_gathered, f_local)
        dist.all_gather(g_gathered, g_local)

        # Re-attach the local computational graph (the gathered copies are detached).
        f_gathered[global_rank] = f_local
        g_gathered[global_rank] = g_local
        return torch.cat(f_gathered, dim=0), torch.cat(g_gathered, dim=0)

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
            # < Efficient Differentiable All-Gather (the SimCLR trick) >
            # Covariance-based losses (LoRA, VAMP, DPNet) need the *global* batch, but
            # their gradient w.r.t. local samples depends only on the local slice. So we
            # all-gather detached copies and re-attach the local differentiable slice,
            # avoiding a redundant gradient all-to-all. The sqrt(svals) scaling is applied
            # to the local slice before gathering (see _assemble_global_batch) so the
            # svals gradient stays consistent with the encoder under DDP's mean-reduce.
            f_x_global, g_y_global = self._assemble_global_batch(
                f_x, g_y, self.trainer.world_size, self.trainer.global_rank
            )

            # svals already applied to the features above -> pass svals=None.
            loss_val = self.loss_fn(f_x_global, g_y_global, svals=None)
            # Scale loss by world size to account for DDP's gradient mean-reduce.
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
            sync_dist=True,
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

    def set_ref_dataset(
        self, dataset: Dataset, loader_kwargs: Dict[str, Any], collate_fn: Callable
    ):
        """
        Stores dataset and config. The actual DataLoader is created in setup()
        to ensure DDP process group is initialized.
        """
        self._ref_dataset = dataset
        self._ref_loader_kwargs = loader_kwargs.copy()
        self._ref_loader_kwargs["collate_fn"] = collate_fn
        # Force these for reference stats
        self._ref_loader_kwargs["shuffle"] = False
        self._ref_loader_kwargs["drop_last"] = False

    def setup(self, stage: str):
        """
        Called by Lightning after DDP process group is initialized.
        We create the ref_loader with DistributedSampler here.
        """
        if self._ref_dataset is None:
            return

        self._ref_max_batches = self._ref_loader_kwargs.pop("max_batches", None)

        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self._ref_dataset, shuffle=False, drop_last=False
            )

        self._ref_loader = DataLoader(
            self._ref_dataset, sampler=sampler, **self._ref_loader_kwargs
        )

    def on_validation_model_eval(self) -> None:
        """
        Executed when the model is switched to eval mode for validation.
        We compute the reference CCA stats here using BN running stats (consistent with analysis).
        """
        super().on_validation_model_eval()

        # Skip during sanity check to save time
        if self.trainer.sanity_checking:
            return

        if self._ref_loader is None:
            return

        # Compute Reference Stats (Distributed)
        self._compute_and_set_ref_stats()

    @torch.inference_mode()
    def _compute_and_set_ref_stats(self) -> None:
        device = self.device
        loader = self._ref_loader
        max_batches = self._ref_max_batches

        if self.trainer.is_global_zero:
            rank_zero_info(
                "[Ref Stats] Accumulating train moments (eval-mode). "
                "This should not take longer than one training epoch."
            )

        # Pre-calculate scaling factor if svals exist
        scale = None
        if self.svals is not None:
            scale = self.svals.view(1, -1).sqrt().detach().to(device=device)

        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = float("inf")

        if max_batches is not None:
            total_batches = min(total_batches, max_batches)

        iterator = iter(loader)

        try:
            first_batch = next(iterator)
        except StopIteration:
            return

        # Helper to update local stats
        x0, y0 = first_batch
        x0 = self._move_batch_to_device(x0, device)
        y0 = self._move_batch_to_device(y0, device)

        # Forward pass (Eval mode is already active)
        f0 = self(x0, lagged=False)
        g0 = self(y0, lagged=True)

        k = f0.shape[1]

        # Accumulators
        M_f = torch.zeros((k, k), dtype=torch.float64, device=device)
        M_g = torch.zeros((k, k), dtype=torch.float64, device=device)
        T_fg = torch.zeros((k, k), dtype=torch.float64, device=device)
        n_samples = torch.zeros((), dtype=torch.float64, device=device)

        # Helper to update local stats
        def accumulate(f, g):
            if scale is not None:
                f = f * scale
                g = g * scale

            f_64 = f.to(torch.float64)
            g_64 = g.to(torch.float64)
            nonlocal M_f, M_g, T_fg, n_samples
            M_f += f_64.T @ f_64
            M_g += g_64.T @ g_64
            T_fg += f_64.T @ g_64
            n_samples += f_64.shape[0]

        # Accumulate first batch
        accumulate(f0, g0)

        # 2. Loop over remaining batches
        for i, batch in enumerate(iterator, start=1):
            if max_batches is not None and i >= max_batches:
                break

            x, y = batch
            x = self._move_batch_to_device(x, device)
            y = self._move_batch_to_device(y, device)

            f = self(x, lagged=False)
            g = self(y, lagged=True)
            accumulate(f, g)

        # 3. All-Reduce (Sum across all GPUs)
        if dist.is_available() and dist.is_initialized():
            # Summing the covariance matrices and the sample count
            dist.all_reduce(M_f, op=dist.ReduceOp.SUM)
            dist.all_reduce(M_g, op=dist.ReduceOp.SUM)
            dist.all_reduce(T_fg, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)

        # 4. Normalize
        if n_samples > 0:
            M_f /= n_samples
            M_g /= n_samples
            T_fg /= n_samples

        # 5. Compute CCA (Each rank computes this identically now)
        # Convert to numpy for linalg compatibility
        stats_np = linalg.OperatorStats(
            M_f=M_f.cpu().numpy(), M_g=M_g.cpu().numpy(), T_fg=T_fg.cpu().numpy()
        )

        try:
            cca_components = linalg.perform_cca(stats_np)
            self.val_metric.set_ref_stats(cca_components)
        except Exception as e:
            if self.trainer.is_global_zero:
                print(f"Warning: Online Reference CCA Failed: {e}")
            self.val_metric.set_ref_stats(None)

    def _move_batch_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(b, device) for b in batch)
        elif isinstance(batch, list):
            return [self._move_batch_to_device(b, device) for b in batch]
        else:
            return batch.to(device)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        f_x = self(x, lagged=False)
        g_y = self(y, lagged=True)

        if self.svals is not None:
            scale = self.svals.sqrt().view(1, -1)
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
