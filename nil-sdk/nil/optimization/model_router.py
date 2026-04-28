"""
ModelRouter — routes subtasks to the optimal model.

Phase 4. Learns which subtasks can be handled by cheaper/faster
models without quality loss, and which need the big model.

Example: Simple classification → Haiku. Complex reasoning → Opus.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nil.optimization.model_router")


class ModelRouter:
    """Routes tasks to the most cost-effective model. Phase 4 stub."""

    def __init__(self, default_model: str = "claude-sonnet-4-20250514") -> None:
        self.default_model = default_model
        self.routing_rules: list[dict[str, Any]] = []

    def route(self, task_type: str, complexity: str = "medium") -> str:
        """Return the best model for a given task type and complexity."""
        # TODO: Learn routing rules from trace analysis
        return self.default_model
