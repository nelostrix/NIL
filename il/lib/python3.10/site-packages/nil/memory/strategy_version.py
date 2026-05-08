"""
StrategyVersion — version control for agent strategies.

When the Evolution Engine mutates a strategy, the old version
is preserved. This enables rollback if the new strategy
performs worse.

Phase 3 feature — stub for now.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StrategyVersion:
    """
    A versioned snapshot of an agent's strategy.

    A strategy includes:
        - System prompt modifications
        - Skill selections
        - Model routing rules
        - Tool ordering preferences
    """

    version: int
    domain: str
    strategy_data: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    is_active: bool = False
    created_at: float = field(default_factory=time.time)
    parent_version: Optional[int] = None
    mutation_type: str = ""  # "skill_addition", "prompt_rewrite", "tool_reorder", etc.
    evaluation_results: dict[str, Any] = field(default_factory=dict)
