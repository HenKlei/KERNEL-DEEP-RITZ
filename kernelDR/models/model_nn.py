import torch
from torch import nn


class _Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def _make_activation(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "softplus":
        return nn.Softplus()
    if name == "sin":
        return _Sin()
    raise ValueError(f"Unknown activation: {name!r}")


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, output_dim, m, depth, activation="tanh"):
        super().__init__()
        self.stack = nn.ModuleList()
        self.stack.append(nn.Linear(input_dim, m))
        for i in range(depth):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, output_dim))
        self.activation_function = _make_activation(activation)

    def forward(self, x):
        for i in range(len(self.stack)-1):
            x = self.activation_function(self.stack[i](x))
        return self.stack[-1](x).flatten()
