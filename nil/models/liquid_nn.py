"""Liquid Neural Network — adaptive dynamics inspired by biological neurons.

Based on "Liquid Time-constant Networks" (Hasani et al., 2021).
The key innovation: each neuron has an adaptive time constant τ that
changes based on input, allowing the network to process temporal data
with 10x fewer parameters than LSTMs.

The differential equation governing each neuron:
    dh/dt = (-h + f(x, h)) / τ(x, h)

where:
    h = hidden state
    x = input
    τ = adaptive time constant (learned, input-dependent)
    f = nonlinear activation function

This makes the network behave like a continuous-time dynamical system
that adapts its speed of processing to the input — fast for rapid
changes, slow for gradual ones. Exactly what robots need.

Standalone usage (NIL):
    from nil.models.liquid_nn import LiquidNN, LiquidCell, LiquidLayer

    model = LiquidNN(input_dim=10, hidden_dim=32, output_dim=5)
    output, hidden = model(x)  # x: [batch, seq_len, input_dim]

With stacc:
    from nil.bridge import NILBridge
    bridge = NILBridge(robot)
    bridge.load("policy")  # loads LNN-based policy

    @robot.on_update
    def control(dt):
        obs = bridge.observe()
        action = bridge.act(obs)
        robot.set_joint_torques(action)

With MAML (few-shot learning):
    from nil.models.liquid_nn import LiquidNN
    from nil.models.meta_learner import MAMLWrapper

    base = LiquidNN(input_dim=4, hidden_dim=32, output_dim=2)
    meta = MAMLWrapper(base, inner_lr=0.01, num_inner_steps=5)
    adapted = meta.adapt(support_x, support_y)
"""

from __future__ import annotations

import math
from re import L
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from nil.models.meta_learner import MAMLWrapper
    from nil.models.world_model import WorldModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    # =================
    # LIQUID CELL - SINGLE layer of Liquid neuron
    # =================

    class LiquidCell(nn.Module):
        """Single Liquid Neural Network cell.

        Implements the ODE:
            dh/dt = (-h + f(W_x @ x + W_h @ h + b)) / τ(x, h)

        where τ is a learned, input-dependent time constant that
        controls how fast the neuron responds.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of liquid neurons
            tau_min: Minimum time constant (fastest response)
            tau_max: Maximum time constant (slowest response)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            tau_min: float = 0.1,
            tau_max: float = 10.0,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.tau_min = tau_min
            self.tau_max = tau_max

            # Input-to-hidden weights
            self.W_x = nn.Linear(input_dim, hidden_dim)
            # hidden-to-hidden weights
            self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

            # Tau network — computes adaptive time constant from input + state
            # τ(x, h) = tau_min + (tau_max - tau_min) * sigmoid(W_tau @ [x, h])
            self.W_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

            self._init_weights()

        def _init_weights(self):
            """Initialize weights for stability."""
            for name, param in self.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.5)

                elif "bias" in name:
                    nn.init.zeros_(param)

        def forward(
            self, x: torch.Tensor, h: torch.Tensor, dt: float = 0.1
        ) -> torch.Tensor:
            """Forward pass - One ODE step.

            Args:
                x:Input [batch, input_dim]
                h:Previous hidden state [batch, hidden_dim]
                dt: Time step for ODE integration

            Returns:
                New hidden state [batch, hidden_dim]
            """
            # concatenate input and hidden for getting
            xh = torch.cat([x, h], dim=-1)

            # Compute adaptive time constant
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
                self.w_tau(xh)
            )

            # Compute pre-activation: f(x, h) = tanh(w_x @ x + W_h @ h)
            f_xh = torch.tanh(self.W_x(x) + self.W_h(h))

            # Sensory gate: How much to let input influence state
            gate = torch.sigmoid(self.W_gate(xh))

            # Gated activation
            f_gated = gate * f_xh + (1 - gate) * h

            # ODE step: dh/dt = (-h + f_gated) / T
            # Euler Integration: h_new = h + dt * dh/dt
            dh = (-h + f_gated) / tau
            h_new = h + dt * dh

            return h_new

        def init_hidden(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
            """Initiate hidden state to zeros."""
            return torch.zeros(batch_size, self.hidden_dim, device=device)

        # ============================
        # LIQUID LAYER - Process a sequence through liquid cells
        # ============================

    class LiquidLayer(nn.Module):
            """A layer of liquid neurons that processes a sequence.

            Takes a sequence of inputs and produces a sequence of hidden states
            by stepping the LiquidCell through time.

            Args:
                input_dim: Input feature dimension
                hidden_dim: Number of liquid neurons
                tau_min: Minimum time constant
                tau_max: Maximum time constant
                bidirectional: Process sequence in both directions
            """

            def __init__(self, input_dim: int, hidden_dim: int,
                        tau_min: float = 0.1, tau_max: float = 10.0,
                        bidirectional: bool = False):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.bidirectional = bidirectional

                self.cell = LiquidCell(input_dim, hidden_dim, tau_min, tau_max)
                if bidirectional:
                    self.cell_bwd = LiquidCell(input_dim, hidden_dim, tau_min, tau_max)
                    self.output_dim = hidden_dim * 2

                else:
                    self.output_dim = hidden_dim

            def forward(
                self,
                x: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                dt: float = 0.1,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                """Process a sequence.

                Args:
                    x: Input sequence [batch, seq_len, input_dim]
                    h0: Initial hidden state [batch, hidden_dim] (optional)
                    dt: Time step for ODE integration

                Returns:
                    (output_sequence, final_hidden_state)
                    output: [batch, seq_len, hidden_dim] (or 2*hidden_dim if bidirectional)
                    hidden: [batch, hidden_dim]
                """
                batch_size, seq_len, _ = x.shape
                device = x.device

                # Initilaize hidden state
                if h0 is None:
                    h = self.cell.init_hidden(batch_size, device)
                else:
                    h = h0

                # Forward pass
                outputs_fwd = []
                for t in range(seq_len):
                    h = self.cell(x[:, t, :], h, dt)
                    outputs_fwd.append(h)

                output_fwd = torch.stack(
                    outputs_fwd, dim=1
                )  # [batch, seq_len, hidden_dim]
                if self.directional:
                    # backward pass
                    h_bwd = self.cell_bwd.init_hidden(batch_size, device)

                    outputs_bwd = []

                    for t in range(seq_len - 1, -1, -1):
                        h_bwd = self.cell_bwd(x[:, t, :], h_bwd, dt)
                        outputs_bwd.append(h_bwd)
                    outputs_bwd.reverse()
                    output_bwd = torch.stack(outputs_bwd, dim=1)

                    output = torch.cat([output_fwd, output_bwd], dim=-1)
                    h_final = torch.cat([h, h_bwd], dim=-1)

                else:
                    output = output_fwd
                    h_final = h

                return output, h_final

            # ==========================
            # LIQUID NEURAL NETWORK  - FULL Model
            # ==========================
    class LiquidNN(nn.Module):
        """Complete Liquid Neural Network.

        Stacks multiple LiquidLayers with optional residual connections,
        layer normalization, and dropout. Output projection maps to
        desired output dimension.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer size per liquid layer
            output_dim: Output dimension
            num_layers: Number of stacked liquid layers
            tau_min: Minimum time constant
            tau_max: Maximum time constant
            dropout: Dropout rate between layers
            bidirectional: Use bidirectional processing
            residual: Use residual connections between layers
            layer_norm: Apply layer normalization
        """

        def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                num_layers: int = 2,
                tau_min: float = 0.1,
                tau_max: float = 10.0,
                dropout: float = 0.1,
                bidirectional: bool = False,
                residual: bool = True,
                layer_norm: bool = True,
            ):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.num_layers = num_layers
                self.residual = residual

                # Build layers
                self.layers = nn.ModuleList()
                self.norm = nn.ModuleList() if layer_norm else None
                self.dropouts = nn.ModuleList()

                # Projection layers for residual connectiions when dims don't match
                self.projections = nn.ModuleList()

                for i in range(num_layers):
                    in_dim = (
                        input_dim
                        if i == 0
                        else (hidden_dim * 2 if bidirectional else hidden_dim)
                    )
                    layer = LiquidLayer(
                        in_dim, hidden_dim, tau_min, tau_max, bidirectional
                    )
                    self.layers.append(layer)

                    layer_out = hidden_dim * 2 if bidirectional else hidden_dim

                    if layer_norm:
                        self.norms.append(nn.LayerNorm(layer_out))

                    self.dropouts.append(nn.Dropout(dropout))

                    # Residual projection if input/output dims differ
                    if residual and in_dim != layer_out:
                        self.projections.append(
                            nn.Linear(in_dim, layer_out, bias=False)
                        )

                    else:
                        self.projections.append(None)

                # Output projections
                final_dim = hidden_dim * 2 if bidirectional else hidden_dim

                self.output_proj = nn.Linear(final_dim, output_dim)

        def forward(
                self,
                x: torch.Tensor,
                h: Optional[List[Torch.Tensor]] = None,
                dt: float = 0.1,
            ) -> Tuple[Torch.Tensor, List[Torch.Tensor]]

            """Forward pass through all layers.

            Args:
                x: Input [batch, seq_len, input_dim]
                h: List of initial hidden states per layer (optional)
                dt: Time step for ODE integration

            Returns:
                (output, hidden_states)
                output: [batch, seq_len, output_dim]
                hidden_states: List of [batch, hidden_dim] per layer
            """
            if h is None:
                h = [None] * self.num_layers

            hidden_states = []
            current = x

            for i, layer in enumerate(self.layers):
                residual_input = current

                output, h_final = layer(current, h[i], dt)
                hidden_states.append(h_final)

                #Layer norm
                if self.norm is not None:
                    output = self.norm[i](output)

                #Residual connection
                if self.residual:
                    proj = self.projections[i]
                    if proj is not None:
                        residual_input = proj(residual_input)
                    output = output + residual_input

                #Dropout
                output = self.dropouts[i](output)
                current = output
            #Output Projection
            output = self.output_proj(current)

            return output, hidden_states

        def predict_step(
            self,
            x: torch.Tensor,
            h: torch.Tensor,
            dt: float = 0.1,
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
            """Single-step prediction (for real-time control).

            Args:
                x: Input [batch, input_dim] (single timestep)
                h: Hidden states from previous step

            Returns:
                (output, hidden_states)
                output: [batch, output_dim]
            """

            #Add sequence dimension
            x_seq = x.unsqueeze(1) # [batch, 1, input_dim]
            output, hidden = self.forward(x_seq, h, dt)
            return output.squeeze(1), hidden

        @property
        def num_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @property
        def num_trainable(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


        def summary(self) -> Dict[str, Any]:
            return {
                "name": "LiquidNN",
                "input_dim": self.input_dim,
                "hidden_dim" : self.hidden_dim,
                "output_dim" : self.output_dim,
                "num_layers": self.num_layers,
                "total_parameters": self.num_parameters,
                "trainable_parameters": self.num_trainable,
            }
else:
    # ══════════════════════════════════════════
    # NUMPY FALLBACK — basic LNN without PyTorch
    # ══════════════════════════════════════════
    class LiquidNN:
        """Minimal LNN implementation using only NumPy.

        for environments without pytorch (edge devices, Raspberry pi).

        Supports inference only - Training requires PyTorch version."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2, **kwargs
        ):
            try:
                import numpy as np
                self._np = np
            except ImportError:
                raise ImportError("Either torch or numpy  required for LiquidNN")

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers

            #Initialize random weights
            self._weights = {}
            in_d = input_dim
            for i in range(num_layers):
                self._weights[f"W_x_{i}"] = np.random.randn(in_d, hidden_dim) * 0.1
                self._weights[f"W_h_{i}"] = np.random.randn(hidden_dim, hidden_dim) * 0.1
                self._weights[f"b_{i}"] = np.zeros(hidden_dim)
                self._weights[f"W_tau_{i}"] = np.random.randn(in_d + hidden_dim, hidden_dim) * 0.1
                in_d = hidden_dim
            self._weights["w_out"] = np.random.randn(hidden_dim, output_dim) * 0.1
            self._weights["b_out"] = np.zeros(output_dim)

            self._hidden = [np.zeros(hidden_dim) for _ in range(num_layers)]


        def predict_step(self, x, dt: float = 0.1):
            """single_step inference."""
            np = self._np
            current = np.array(x, dtype=float)

            for i in range(self.num_layers):
                h = self._hidden[i]
                xh = np.concatenate([current, h])

                # Adaptive time constant
                tau_raw = xh @ self._weights[f"W_tau_{i}"]
                tau = 0.1 + 9.9 / (1 + np.exp(-tau_raw))#Sigmoid Scaled

                #pre-activation
                f_xh = np.tanh(current @ self._weights[f"W_x_{i}"] + h @ self._weights[f"W_h_{i}"] + self._weights[f"b_{i}"])

                #ODE step
                dh = (-h + f_xh) / tau
                self._hidden[i] = h + dt * dh
                current = self._hidden[i]

            #output
            output = current @ self._weights["W_out"] + self._weights["b_out"]
            return output
        def load_weights(self, path: str):
            """Load weights from a .npz file (Explored form Pytorch version)."""
            np = self._np
            data = np.load(path)
            for key in data:
                if key in self._weights:
                    self._weights[key] = data[key]

        def reset(self):
            self._hidden = [self._np.zeros(self.hidden_dim) for _ in range(self.num_layers)]

        @property
        def num_parameters(self) -> int:
            return sum(w.size for w in self._weights.values())

        def summary(self) -> Dict[str, Any]:
            return {
                "name" : "LiquidNN (Numpy fallback)",
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers" : self.num_layers,
                "total_params": self.num_parameters,
                "backend": "numpy",
            }
    class MAMLWrapper:
        """Placeholder  - MAML requires PyTorch."""
        def __init__(self, *args, **kwargs):
            raise ImportError ("MAMLWrapper requires Pytorch, Install: pip install torch")
    class Worldmodel:
        """placeHolder - WorldModel requires PyTorch."""
        def __init__(self, *args, **kwargs):
            raise ImportError("WorldModel requires PyTorch. Install: pip install torch")
    def create_liquid_policy(*args, **kwargs):
        raise ImportError("Create_liquid_policy requires PyTorch. install >> pip install torch")
    def create_world_model(*args, **kwargs):
        raise ImportError("Create_world_model requires Pytorch. install>> pip install torch")
