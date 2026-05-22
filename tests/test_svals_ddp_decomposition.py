"""Regression for the learned-singular-value gradient under the DDP step.

KoopmanPLModule.training_step (world_size > 1) gathers features across ranks,
keeps only the local slice differentiable, scales it by sqrt(svals) *before*
gathering (KoopmanPLModule._assemble_global_batch), and multiplies the loss by
world_size; DDP then mean-reduces the gradients. This must reproduce a
single-process full-batch gradient for BOTH the encoder and the learned svals.

The earlier bug scaled svals *inside* the loss on the gathered batch, so every
rank computed the full-batch svals gradient and it ended up inflated by
world_size. These tests reproduce the DDP rank-decomposition with autograd in a
single process (no NCCL needed), so they run in CI on CPU.
"""

import torch

from koopmansvd.models.encoders.mlp import MLPEncoder
from koopmansvd.models.lightning import KoopmanPLModule
from koopmansvd.models.losses import NestedLoRALoss

IN_DIM = 6
N_MODES = 4
WORLD = 2
PER = 8


def _module():
    torch.manual_seed(0)
    enc = MLPEncoder(input_dim=IN_DIM, output_dim=N_MODES, hidden_dims=[16])
    module = KoopmanPLModule(
        encoder=enc,
        loss_fn=NestedLoRALoss(n_modes=N_MODES, nesting="jnt"),
        learn_svals=True,
        num_svals=N_MODES,
    )
    return module.double()


def _full_batch_grads(module, x, y, params):
    f = module(x, lagged=False)
    g = module(y, lagged=True)
    s = module.svals.view(1, -1).sqrt()
    loss = module.loss_fn(f * s, g * s, svals=None)
    return torch.autograd.grad(loss, params)


def _ddp_decomposed_grads(module, x, y, params, *, scale_inside_loss):
    """Sum, over simulated ranks, the gradient of (loss_global * world) flowing
    through each rank's local slice, then divide by world (DDP mean-reduce).

    scale_inside_loss=False reproduces the fix (svals applied to the local slice
    before gathering); True reproduces the old buggy behaviour.
    """
    accum = [torch.zeros_like(p) for p in params]
    for r in range(WORLD):
        f = module(x, lagged=False)
        g = module(y, lagged=True)
        s = module.svals.view(1, -1).sqrt()
        f_parts, g_parts = [], []
        for rr in range(WORLD):
            sl = slice(rr * PER, (rr + 1) * PER)
            fr, gr = f[sl], g[sl]
            if not scale_inside_loss:
                fr, gr = fr * s, gr * s
            if rr != r:  # remote slices are detached constants
                fr, gr = fr.detach(), gr.detach()
            f_parts.append(fr)
            g_parts.append(gr)
        svals_arg = module.svals if scale_inside_loss else None
        loss_r = module.loss_fn(torch.cat(f_parts), torch.cat(g_parts), svals=svals_arg)
        loss_r = loss_r * WORLD
        for i, gr in enumerate(torch.autograd.grad(loss_r, params)):
            accum[i] = accum[i] + gr
    return [a / WORLD for a in accum]


def test_ddp_step_matches_full_batch_gradients():
    module = _module()
    names = [n for n, _ in module.named_parameters()]
    params = [p for _, p in module.named_parameters()]

    x = torch.randn(WORLD * PER, IN_DIM, dtype=torch.float64)
    y = torch.randn(WORLD * PER, IN_DIM, dtype=torch.float64)

    ref = _full_batch_grads(module, x, y, params)
    ddp = _ddp_decomposed_grads(module, x, y, params, scale_inside_loss=False)

    for name, d, r in zip(names, ddp, ref):
        assert torch.allclose(d, r, rtol=1e-6, atol=1e-9), (
            f"{name}: DDP gradient != full-batch gradient "
            f"(max|delta|={(d - r).abs().max().item():.3e})"
        )


def test_old_scaling_inside_loss_inflates_svals_gradient():
    """Characterize the bug: scaling svals inside the loss leaves the encoder
    gradient correct but inflates the svals gradient by exactly world_size."""
    module = _module()
    named = list(module.named_parameters())
    params = [p for _, p in named]
    svals_i = next(i for i, (n, _) in enumerate(named) if "svals" in n)

    x = torch.randn(WORLD * PER, IN_DIM, dtype=torch.float64)
    y = torch.randn(WORLD * PER, IN_DIM, dtype=torch.float64)

    ref = _full_batch_grads(module, x, y, params)
    buggy = _ddp_decomposed_grads(module, x, y, params, scale_inside_loss=True)

    # Encoder gradients stay correct even with the bug...
    for i, (name, _) in enumerate(named):
        if i == svals_i:
            continue
        assert torch.allclose(buggy[i], ref[i], rtol=1e-6, atol=1e-9), name
    # ...but the svals gradient is inflated by world_size.
    assert not torch.allclose(buggy[svals_i], ref[svals_i], rtol=1e-3, atol=1e-6)
    assert torch.allclose(buggy[svals_i], WORLD * ref[svals_i], rtol=1e-5, atol=1e-7)
