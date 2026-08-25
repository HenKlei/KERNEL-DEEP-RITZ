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
        # Kept so that a checkpoint can be rebuilt without knowing the command
        # line that produced it, see save_nn_model / load_nn_model.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m
        self.depth = depth
        self.activation = activation
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


def save_nn_model(model, path):
    """Persist a network together with everything needed to rebuild it.

    Storing the architecture alongside the weights makes the checkpoint
    self-contained, so that the error metrics of a trained network can be
    recomputed later -- on a different evaluation grid, for instance -- without
    repeating the training.
    """
    torch.save({
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "m": model.m,
        "depth": model.depth,
        "activation": model.activation,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_nn_model(path, device=None):
    """Rebuild a network saved by save_nn_model."""
    checkpoint = torch.load(path, map_location="cpu")
    model = NeuralNetworkModel(checkpoint["input_dim"], checkpoint["output_dim"],
                               checkpoint["m"], checkpoint["depth"],
                               activation=checkpoint["activation"])
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model = model.to(device)
    return model
