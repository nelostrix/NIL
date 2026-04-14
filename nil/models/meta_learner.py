"""
Model-Agnostic Meta-Learning (MAML) wrapper.
Based on: "Model-Agnostic Meta-Learning" (Finn et al., 2017)
"""

from copy import deepcopy

# from modulefinder import Module
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.stateless import functional_call


class MAMLWrapper(nn.Module):
    """Model-Agnostic Meta-Learning wrapper for any PyTorch model.

    Implements the MAML algorithm (Finn et al., 2017):
    - Outer loop: learn an initialization that enables fast adaptation
    - Inner loop: adapt to a new task with a few gradient steps

    The result: a model that can learn new tasks from <10 examples.

    Args:
        model: Base model (typically a LiquidNN)
        inner_lr: Learning rate for inner loop adaptation
        num_inner_steps: Number of gradient steps for adaptation
        first_order: Use first-order approximation (faster, less memory)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        first_order: bool = False,
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Adapt the model to a new task using support examples.

        Args:
            support_x: Support set inputs [num_examples, ...]
            support_y: Support set targets [num_examples, ...]
            loss_fn: Loss function (default: MSE)

        Returns:
            Adapted copy of the model
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Cloned Model parameters
        adapted_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }

        for _ in range(self.num_inner_steps):
            output = self._forward_with_params(support_x, adapted_params)

            if output.dim() == 3:
                output = output[:, -1, :]

            loss = loss_fn(output, support_y)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )
            adapted_params = {
                name: param
                - self.inner_lr
                * (grad if grad is not None else torch.zeros_like(param))
                for (name, param), grad in zip(adapted_params.items(), grads)
            }
        return adapted_params

    def _forward_with_params(self, x, params):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        output, _ = functional_call(self.model, params, (x,))
        return output

    def meta_loss(self, task_batch, loss_fn=None):
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0

        for support_x, support_y, query_x, query_y in task_batch:
            adapted_params = self.adapt(support_x, support_y, loss_fn)

            if query_x.dim() == 2:
                query_x = query_x.unsqueeze(1)

            output = self._forward_with_params(query_x, adapted_params)

            if output.dim() == 3:
                output = output[:, -1, :]

            task_loss = loss_fn(output, query_y)
            total_loss += task_loss

        return total_loss / len(task_batch)
