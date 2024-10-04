from torch import nn


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, output_dim, m, depth):
        super().__init__()
        self.stack = nn.ModuleList()
        self.stack.append(nn.Linear(input_dim, m))
        for i in range(depth):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, output_dim))
        self.activation_function = nn.Tanh()

    def forward(self, x):
        for i in range(len(self.stack)-1):
            x = self.activation_function(self.stack[i](x))
        return self.stack[-1](x).flatten()
