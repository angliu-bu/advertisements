"""Multi-Layer Perceptron model for advertisement CTR prediction.

Example:
    >>> import torch
    >>> from .model_mlp import MLP
    >>> model = MLP(input_dim=20, hidden_dims=[64, 32], activation='relu', dropout=0.2)
    >>> x = torch.randn(5, 20)
    >>> logits = model(x)
"""

from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):
    """Configurable multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        hidden_dims = list(hidden_dims or [64, 32])
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # pragma: no cover - simple wrapper
        return self.network(x)
