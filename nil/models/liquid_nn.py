"""
Liquid Neural Network implementation.
Based on: "Liquid Time-constant Networks" (Hasani et al., 2021)
"""

import torch
import torch.nn as nn


class LiquidNeuron(nn.Module):
    """Single liquid neuron with adaptive time constants."""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input weights
        self.W_x = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent weights
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        
        # Time constant network (learns adaptive dynamics)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, h, dt=0.1):
        """
        Args:
            x: Input [batch, input_dim]
            h: Hidden state [batch, hidden_dim]
            dt: Time step (default 0.1s)
        Returns:
            h_next: Next hidden state
        """
        # Compute adaptive time constants
        tau = self.tau_net(torch.cat([x, h], dim=-1))
        
        # Input contribution
        input_contrib = self.W_x(x)
        
        # Recurrent contribution
        recurrent_contrib = self.W_h(h)
        
        # Liquid dynamics: dh/dt = (-h + f(x, h)) / tau
        f_xh = torch.tanh(input_contrib + recurrent_contrib)
        dh_dt = (-h + f_xh) / (tau + 1e-6)
        h_next = h + dt * dh_dt
        
        return h_next


class LiquidNN(nn.Module):
    """Multi-layer Liquid Neural Network."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Stack of liquid neurons
        self.layers = nn.ModuleList([
            LiquidNeuron(
                input_dim if i == 0 else hidden_dim,
                hidden_dim
            )
            for i in range(num_layers)
        ])
        
        # Output layer
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
            h = [torch.zeros(batch_size, self.hidden_dim, device=x.device)
                 for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Pass through liquid layers
            for i, layer in enumerate(self.layers):
                h[i] = layer(x_t if i == 0 else h[i-1], h[i], dt)
            
            # Output
            y_t = self.output_layer(h[-1])
            outputs.append(y_t)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs, h


if __name__ == "__main__":
    # Test
    model = LiquidNN(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2)
    x = torch.randn(4, 50, 10)
    outputs, h = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Num parameters: {sum(p.numel() for p in model.parameters()):,}")
