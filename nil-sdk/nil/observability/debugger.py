from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DebugCommandType(str, Enum):
    CONTINUE = "continue"
    STEP = "step"
    INSPECT = "inspect"
    EDIT_PROMPT = "edit_prompt"
    SET_BREAKPOINT = "set_breakpoint"
    SNAPSHOT = "snapshot"
    ROLLBACK = "rollback"


@dataclass
class DebugCommand:
    type: DebugCommandType = DebugCommandType.CONTINUE
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Breakpoint:
    condition: str = ""
    step_number: Optional[int] = None
    tool_name: Optional[str] = None
    event_type: Optional[str] = None

    def should_break(self, step: int, tool: str = "", event: str = "") -> bool:
        if self.step_number is not None and step >= self.step_number:
            return True
        if self.tool_name is not None and tool == self.tool_name:
            return True
        if self.event_type is not None and event == self.event_type:
            return True
        return False


@dataclass
class AgentSnapshot:
    run_id: str = ""
    step_number: int = 0
    system_prompt: str = ""
    messages: list[dict] = field(default_factory=list)
    memory_state: dict[str, Any] = field(default_factory=dict)
    pending_tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_number": self.step_number,
            "system_prompt": self.system_prompt[:500],
            "messages": list(self.messages)[-5:],
            "memory_state": dict(self.memory_state),
            "pending_tool_calls": list(self.pending_tool_calls),
        }


class LiveDebugger:
    def __init__(self):
        self.paused = False
        self.breakpoints: list[Breakpoint] = []
        self.snapshot_history: list[AgentSnapshot] = []
        self._pending_command: Optional[DebugCommand] = None

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._pending_command = DebugCommand(type=DebugCommandType.CONTINUE)

    def step(self) -> None:
        self._pending_command = DebugCommand(type=DebugCommandType.STEP)

    def set_breakpoint(self, breakpoint: Breakpoint) -> None:
        self.breakpoints.append(breakpoint)

    def clear_breakpoints(self) -> None:
        self.breakpoints.clear()

    def should_pause(self, step: int, tool: str = "", event: str = "") -> bool:
        if self.paused:
            return True
        for bp in self.breakpoints:
            if bp.should_break(step, tool, event):
                return True
        return False

    def take_snapshot(
        self,
        run_id: str,
        step_number: int,
        system_prompt: str = "",
        messages: Optional[list[dict]] = None,
        memory_state: Optional[dict] = None,
        pending_tool_calls: Optional[list[dict]] = None,
    ) -> AgentSnapshot:
        snapshot = AgentSnapshot(
            run_id=run_id,
            step_number=step_number,
            system_prompt=system_prompt,
            messages=messages or [],
            memory_state=memory_state or {},
            pending_tool_calls=pending_tool_calls or [],
        )
        self.snapshot_history.append(snapshot)
        return snapshot

    def rollback(self) -> Optional[AgentSnapshot]:
        if len(self.snapshot_history) >= 2:
            return self.snapshot_history[-2]
        return None

    def execute_command(self, command: DebugCommand) -> str:
        if command.type == DebugCommandType.CONTINUE:
            self.resume()
            return "Resumed execution"
        elif command.type == DebugCommandType.STEP:
            self.step()
            return "Stepping to next step"
        elif command.type == DebugCommandType.INSPECT:
            return f"Inspecting: {command.payload}"
        elif command.type == DebugCommandType.EDIT_PROMPT:
            return f"Prompt updated: {command.payload.get('prompt', '')[:100]}"
        elif command.type == DebugCommandType.SET_BREAKPOINT:
            bp = Breakpoint(**command.payload)
            self.set_breakpoint(bp)
            return f"Breakpoint set: {bp.condition}"
        elif command.type == DebugCommandType.SNAPSHOT:
            snap = self.snapshot_history[-1] if self.snapshot_history else None
            return f"Snapshot v{len(self.snapshot_history)}: step {snap.step_number if snap else 'N/A'}"
        elif command.type == DebugCommandType.ROLLBACK:
            snap = self.rollback()
            return f"Rolled back to step {snap.step_number}" if snap else "Nothing to roll back to"
        return f"Unknown command: {command.type}"
