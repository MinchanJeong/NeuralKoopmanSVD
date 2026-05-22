import pytest
import torch
import torch.nn as nn

from koopmansvd.models.encoders.mlp import MLPEncoder


@pytest.mark.parametrize(
    "activation",
    ["ReLU", nn.GELU, nn.Tanh()],
    ids=["str", "class", "instance"],
)
def test_mlp_accepts_activation_spec(activation):
    """MLPEncoder must accept the activation as a name (str), a class, or a
    pre-built module instance. Passing an instance (e.g. nn.Tanh()) previously
    raised TypeError because the fallback factory was called with no argument.
    """
    enc = MLPEncoder(
        input_dim=4, output_dim=3, hidden_dims=[8, 8], activation=activation
    )
    out = enc(torch.randn(5, 4))
    assert out.shape == (5, 3)


def test_mlp_rejects_unknown_activation_string():
    with pytest.raises(ValueError):
        MLPEncoder(input_dim=4, output_dim=3, activation="NotAnActivation")
