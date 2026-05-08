"""
BudgetController — enforces cost limits per run and per domain.

Phase 4. Prevents runaway costs by tracking token usage and
stopping execution when budgets are exceeded.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nil.optimization.budget_controller")


class BudgetController:
    """Tracks and enforces cost budgets. Phase 4 stub."""

    def __init__(
        self,
        max_cost_per_run: float = 1.0,
        max_cost_per_day: float = 50.0,
    ) -> None:
        self.max_cost_per_run = max_cost_per_run
        self.max_cost_per_day = max_cost_per_day
        self._daily_spend: float = 0.0

    def check(self, current_run_cost: float) -> bool:
        """Return True if within budget, False if should stop."""
        if current_run_cost > self.max_cost_per_run:
            logger.warning(
                "Run cost $%.4f exceeds limit $%.2f",
                current_run_cost,
                self.max_cost_per_run,
            )
            return False
        if self._daily_spend + current_run_cost > self.max_cost_per_day:
            logger.warning("Daily spend would exceed $%.2f", self.max_cost_per_day)
            return False
        return True

    def record_spend(self, amount: float) -> None:
        self._daily_spend += amount
