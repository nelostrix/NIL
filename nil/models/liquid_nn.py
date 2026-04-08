from doctest import OutputChecker

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidNeuron(nn.Module):
    """single liquid neuron with adaptive time constraints."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input.dim = input_dim
        self.hidden.dim = hidden_dim

        # input weights
        self.W_x = nn.Linear(input_dim, hidden_dim)

        # recurrent weights
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

        # Time constant network (learns adaptive dynamics)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, hidden_dim), nn.sigmoid()
        )

    def forward(self, x, h, dt=0.1):
        """args:
            x: input [batch, input_dim]
            h: hidden state [batch, hidden_dim]
            dt: Time_Step (defualt 0.1sec)
        returns:
            h_next: Next Hidden state"""

        # Compute adaptive time constants
        tau = self.tau_net(torch.cat([x, h], dim=-1))

        # input Contribution
        input_contrib = self.W_x(x)

        # recurrent Contrib
        recurrent_contrib = self.W_h(h)

        # Liquid dynamics: dh/dt = (-h + (f(x, h)))/tau
        # Discrete approximation: h_next = h + dt * dh/dt
        f_xh = torch.tanh(input_contrib + recurrent_contrib)
        dh_dt = (-h + f_xh) / (tau + 1e-6)
        h_next = h + dt * dh_dt

        return h_next


class LiquidNN(nn.Module):
    """Multi-Layer Liquid Neural Network."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Stack of Liquid neurons
        self.layers = nn.ModuleList(
            [
                LiquidNeuron(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )
        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h=None, dt=0.1):
        """
        Args:
            x: Input sequence [batch, seq_len, input_dim]
            h: Initial hidden states (optional)
            dt: Time step
        Returns:
            outputs: Predictions [batch, seq_len, output_dim]
            h_final: Final hidden states
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states
        if h is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        outputs = []

        # process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]

        for i, layer in enumerate(self.layers):
            h[i] = layer(x_t if i == 0 else h[i - 1], h[i], dt)

        outputs = torch.stack(outputs, dim=1)
        return outputs, h

    def get_params(self):
        """Get trainable parameters (for meta-learning)."""
        return list(self.parameters())


if __name__ == "__main__":
    # test
    model = LiquidNN(input_dim=10, hidden_dim=32, output_dim=5, num_layers=4)
    x = torch.randn(4, 50, 10)  # [batch=4, seq=50, features=10]
    outputs, h = model(x)
    print(f"input_shape: {x.shape}")
    print(f"Output_shape: {outputs.shape}")
    print(f"Num parameters: {sum(p.numel() for p in model.parameters()):,}")
