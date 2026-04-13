"""
Model-Agnostic Meta-Learning (MAML) wrapper.
Based on: "Model-Agnostic Meta-Learning" (Finn et al., 2017)
"""

from copy import deepcopy

import torch
import torch.nn as nn


class MAMLWrapper(nn.Module):
    """MAML wrapper for any PyTorch model."""

    def __init__(self, model, inner_lr=0.01, num_inner_steps=5):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

    def adapt(self, support_x, support_y, loss_fn=None):
        """
        Fast adaptation on support set (inner loop).

        Args:
            support_x: Support inputs [batch, seq, input_dim]
            support_y: Support targets [batch, seq, output_dim]
            loss_fn: Loss function (default: MSE)
        Returns:
            adapted_model: Model with updated parameters
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Clone model for adaptation (doesn't affect original)
        adapted_model = deepcopy(self.model)

        # Inner loop optimization
        for step in range(self.num_inner_steps):
            # Forward pass
            pred, _ = adapted_model(support_x)
            loss = loss_fn(pred, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=True,  # For meta-gradient
            )

            # Manual parameter update (SGD)
            with torch.no_grad():
                for param, grad in zip(adapted_model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad

        return adapted_model

    def meta_loss(self, task_batch, loss_fn=None):
        """
        Compute meta-loss across task batch (outer loop).

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y)
            loss_fn: Loss function
        Returns:
            meta_loss: Average loss on query sets
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        meta_losses = []  # ← PLURAL - this is the list that collects losses

        for support_x, support_y, query_x, query_y in task_batch:
            # Adapt to task
            adapted_model = self.adapt(support_x, support_y, loss_fn)

            # Evaluate on query set
            query_pred, _ = adapted_model(query_x)
            query_loss = loss_fn(query_pred, query_y)

            meta_losses.append(query_loss)  # ← PLURAL - append to the list

        # Average across tasks
        meta_loss = torch.stack(meta_losses).mean()  # ← SINGULAR - final averaged loss

        return meta_loss

    def forward(self, x):
        """Forward pass through base model."""
        return self.model(x)


if __name__ == "__main__":
    from liquid_nn import LiquidNN

    # Create base model
    base_model = LiquidNN(input_dim=10, hidden_dim=32, output_dim=5)

    # Wrap with MAML
    meta_learner = MAMLWrapper(base_model, inner_lr=0.01, num_inner_steps=5)

    # Simulate task
    support_x = torch.randn(2, 20, 10)
    support_y = torch.randn(2, 20, 5)
    query_x = torch.randn(2, 20, 10)
    query_y = torch.randn(2, 20, 5)

    # Adapt to task
    adapted_model = meta_learner.adapt(support_x, support_y)

    # Test adapted model
    pred, _ = adapted_model(query_x)
    print(f"Query prediction shape: {pred.shape}")
    print("✓ MAMLWrapper works!")
