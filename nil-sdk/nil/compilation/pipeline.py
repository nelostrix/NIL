from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("nil.compilation")


class CompilationLevel(Enum):
    TEXT_FRAGMENT = 0
    FEW_SHOT = 1
    CONSTRAINED = 2
    DSL_SCRIPT = 3
    NATIVE_CODE = 4
    COMPILED_TREE = 5


@dataclass
class ControlFlowNode:
    node_type: str = "tool_call"
    tool_name: Optional[str] = None
    condition: Optional[str] = None
    children: Optional[list[ControlFlowNode]] = None
    subskill_name: Optional[str] = None


@dataclass
class IntermediateRepresentation:
    name: str = ""
    tools: list[str] = field(default_factory=list)
    control_flow: list[ControlFlowNode] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "parsed"
    is_linear: bool = True
    has_conditionals: bool = False
    estimated_steps: int = 0
    can_compile: bool = False

    def __post_init__(self):
        self.is_linear = not self.has_conditionals and not self.control_flow
        self.estimated_steps = len(self.tools)
        self.can_compile = bool(self.tools) and self.is_linear

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "confidence": self.confidence,
            "source": self.source,
            "is_linear": self.is_linear,
            "has_conditionals": self.has_conditionals,
            "estimated_steps": self.estimated_steps,
            "can_compile": self.can_compile,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntermediateRepresentation:
        return cls(
            name=d.get("name", ""),
            tools=list(d.get("tools", [])),
            preconditions=list(d.get("preconditions", [])),
            postconditions=list(d.get("postconditions", [])),
            confidence=d.get("confidence", 0.0),
            source=d.get("source", "parsed"),
            has_conditionals=d.get("has_conditionals", False),
        )


@dataclass
class CompilationResult:
    skill_name: str = ""
    level: CompilationLevel = CompilationLevel.TEXT_FRAGMENT
    compiled: bool = False
    compiled_object: Any = None
    ir: Optional[IntermediateRepresentation] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "level": self.level.value,
            "compiled": self.compiled,
            "ir": self.ir.to_dict() if self.ir else None,
        }


class _SkillStub:
    def __init__(self, name: str = "", description: str = "", action_sequence: Optional[list[str]] = None,
                 preconditions: Optional[list[str]] = None, postconditions: Optional[list[str]] = None,
                 confidence: float = 0.0, usage_count: int = 0, tool_names: Optional[list[str]] = None):
        self.name = name
        self.description = description
        self.action_sequence = action_sequence or []
        self.preconditions = preconditions or []
        self.postconditions = postconditions or []
        self.confidence = confidence
        self.usage_count = usage_count
        self.tool_names = tool_names or []


class SkillParser:
    _KNOWN_TOOLS_RE = re.compile(r'\b(search_logs|query_metrics|analyze_logs|check_db|restart_service|drain_connections|deploy|rollback|scale_up|slack_notify|pagerduty_alert|check_endpoint|diagnose|investigate)\b', re.IGNORECASE)

    def parse(self, skill: _SkillStub) -> IntermediateRepresentation:
        tools = self._extract_tool_sequence(skill)
        control_flow = self._extract_control_flow(skill.description)
        pre = skill.preconditions or []
        post = skill.postconditions or []

        return IntermediateRepresentation(
            name=skill.name,
            tools=tools,
            control_flow=control_flow,
            preconditions=pre,
            postconditions=post,
            confidence=skill.confidence,
            source="parsed" if tools else "text_only",
        )

    def _extract_tool_sequence(self, skill: _SkillStub) -> list[str]:
        if skill.action_sequence:
            return skill.action_sequence
        return self._find_tool_names(skill.description)

    def _find_tool_names(self, text: str) -> list[str]:
        matches = self._KNOWN_TOOLS_RE.findall(text)
        seen: list[str] = []
        for m in matches:
            if m.lower() not in [s.lower() for s in seen]:
                seen.append(m.lower())
        return seen

    def _extract_control_flow(self, description: str) -> list[ControlFlowNode]:
        nodes = []
        desc_lower = description.lower()
        if "if " in desc_lower or "when " in desc_lower:
            nodes.append(ControlFlowNode(
                node_type="condition",
                condition=self._extract_condition(description),
            ))
        if "parallel" in desc_lower or "concurrent" in desc_lower or "simultaneous" in desc_lower:
            nodes.append(ControlFlowNode(node_type="parallel"))
        if "retry" in desc_lower or "loop" in desc_lower:
            nodes.append(ControlFlowNode(node_type="loop", condition="retry"))
        return nodes

    def _extract_condition(self, description: str) -> str:
        match = re.search(r'(?:if|when)\s+(.+?)(?:,|\.|$)', description, re.IGNORECASE)
        return match.group(1).strip() if match else ""


class SkillOptimizer:
    def __init__(self):
        self.fusion_map: dict[tuple[str, str], str] = {}
        self.specialization_map: dict[str, str] = {}

    def register_fusion(self, first: str, second: str, fused_name: str) -> None:
        self.fusion_map[(first, second)] = fused_name

    def register_specialization(self, generic: str, specialized: str) -> None:
        self.specialization_map[generic] = specialized

    def optimize(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        if ir.is_linear:
            ir = self._eliminate_dead_steps(ir)
            ir = self._fuse_adjacent_pairs(ir)
        ir = self._specialize(ir)
        return ir

    def _eliminate_dead_steps(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        if len(ir.tools) < 2:
            return ir
        alive = [ir.tools[0]]
        for i in range(1, len(ir.tools)):
            prev = ir.tools[i - 1]
            if self._output_consumed_by(prev, ir.tools[i] if i < len(ir.tools) else prev):
                alive.append(prev)
        alive.append(ir.tools[-1])
        unique = [alive[0]]
        for t in alive[1:]:
            if t != unique[-1]:
                unique.append(t)
        ir.tools = unique
        return ir

    def _fuse_adjacent_pairs(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        fused = []
        i = 0
        while i < len(ir.tools):
            if i + 1 < len(ir.tools):
                pair = (ir.tools[i], ir.tools[i + 1])
                if pair in self.fusion_map:
                    fused.append(self.fusion_map[pair])
                    i += 2
                    continue
            fused.append(ir.tools[i])
            i += 1
        ir.tools = fused
        return ir

    def _specialize(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        specialized = []
        for tool_name in ir.tools:
            specialized.append(self.specialization_map.get(tool_name, tool_name))
        ir.tools = specialized
        return ir

    def _output_consumed_by(self, prev_tool: str, curr_tool: str) -> bool:
        return True


class SkillCodeGenerator:
    def __init__(self):
        self._tool_descriptions: dict[str, str] = {}
        self._tool_params: dict[str, dict] = {}

    def register_tool_info(self, name: str, description: str, parameters: Optional[dict] = None) -> None:
        self._tool_descriptions[name] = description
        self._tool_params[name] = parameters or {}

    def generate_python(self, ir: IntermediateRepresentation) -> str:
        lines = [
            f"def compiled_{ir.name}(tool_manager, **kwargs):",
            f'    """Compiled skill: {ir.name}"""',
            "    results = {}",
            "",
        ]
        for i, tool_name in enumerate(ir.tools):
            var_name = f"result_{i}"
            lines.append(f"    # Step {i + 1}: {tool_name}")
            lines.append(f"    {var_name} = tool_manager.execute(")
            lines.append(f"        '{tool_name}', kwargs")
            lines.append(f"    )")
            lines.append(f"    if {var_name}.get('error'):")
            lines.append(f"        return {{'error': {var_name}['error'], 'step': {i}}}")
            lines.append(f"    results['{tool_name}'] = {var_name}.get('output')")
            lines.append("")
        lines.append("    return {'output': results, 'error': None}")
        return "\n".join(lines)

    def generate_constrained_grammar(self, ir: IntermediateRepresentation) -> dict:
        steps = []
        for tool_name in ir.tools:
            desc = self._tool_descriptions.get(tool_name, "")
            params = self._tool_params.get(tool_name, {})
            steps.append({
                "step_type": "tool_call",
                "tool_name": tool_name,
                "description": desc,
                "parameters": {
                    k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                    for k, v in params.items()
                },
            })
        return {
            "skill_name": ir.name,
            "type": "constrained_sequence",
            "steps": steps,
            "preconditions": ir.preconditions,
            "postconditions": ir.postconditions,
        }

    def generate_dsl(self, ir: IntermediateRepresentation) -> str:
        lines = [f"SKILL {ir.name}"]
        if ir.preconditions:
            lines.append(f"  PRECONDITIONS: {', '.join(ir.preconditions)}")
        if ir.postconditions:
            lines.append(f"  POSTCONDITIONS: {', '.join(ir.postconditions)}")
        for i, tool_name in enumerate(ir.tools):
            lines.append(f"  STEP {i + 1}: CALL {tool_name}")
            lines.append(f"    ON_ERROR: fail")
        lines.append("  OUTPUT: last_result")
        return "\n".join(lines)


class _ToolManagerStub:
    def execute(self, tool_name: str, kwargs: dict) -> dict:
        return {"output": f"{tool_name} executed", "error": None}


class CompiledSkillExecutor:
    def __init__(self):
        self.parser = SkillParser()
        self.optimizer = SkillOptimizer()
        self.codegen = SkillCodeGenerator()
        self.tool_manager: Any = _ToolManagerStub()
        self._compiled_cache: dict[str, Callable] = {}

    def compile_and_cache(self, skill: _SkillStub, level: CompilationLevel) -> Any:
        ir = self.parser.parse(skill)
        ir = self.optimizer.optimize(ir)

        if level == CompilationLevel.NATIVE_CODE:
            code = self.codegen.generate_python(ir)
            compiled = self._exec_to_function(code, f"compiled_{skill.name}")
            self._compiled_cache[skill.name] = compiled
            return compiled

        elif level == CompilationLevel.DSL_SCRIPT:
            return self.codegen.generate_dsl(ir)

        elif level == CompilationLevel.CONSTRAINED:
            return self.codegen.generate_constrained_grammar(ir)

        return None

    def execute(self, skill_name: str, context: Optional[dict] = None) -> Any:
        compiled = self._compiled_cache.get(skill_name)
        if compiled:
            return compiled(self.tool_manager, **(context or {}))
        raise ValueError(f"Skill '{skill_name}' not compiled")

    def _exec_to_function(self, code: str, func_name: str) -> Callable:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        return namespace.get(func_name, lambda *a, **kw: {"error": f"Function {func_name} not found"})


class CompilationLearner:
    def __init__(self):
        self.min_uses_for_compilation: int = 3
        self._trace_data: dict[str, list[dict]] = {}

    def add_trace_for_skill(self, skill_name: str, tool_sequence: list[str], success: bool = True) -> None:
        if skill_name not in self._trace_data:
            self._trace_data[skill_name] = []
        self._trace_data[skill_name].append({"tool_sequence": tool_sequence, "success": success})

    def should_compile(self, skill: _SkillStub) -> tuple[bool, CompilationLevel]:
        if skill.usage_count < self.min_uses_for_compilation:
            return False, CompilationLevel.TEXT_FRAGMENT

        determinism = self._measure_determinism(skill)

        if determinism > 0.95 and skill.confidence > 0.9:
            return True, CompilationLevel.NATIVE_CODE
        elif determinism > 0.8 and skill.confidence > 0.8:
            return True, CompilationLevel.DSL_SCRIPT
        elif determinism > 0.6:
            return True, CompilationLevel.CONSTRAINED
        else:
            return False, CompilationLevel.TEXT_FRAGMENT

    def _measure_determinism(self, skill: _SkillStub) -> float:
        traces = self._trace_data.get(skill.name, [])
        if len(traces) < 2:
            return 0.0

        sequences = [t["tool_sequence"] for t in traces]
        first_seq = sequences[0]
        matches = sum(1 for s in sequences if s == first_seq)
        sequence_consistency = matches / len(sequences)
        return sequence_consistency


class SkillMacro:
    def __init__(self, skill: _SkillStub):
        self.name = skill.name
        self.steps = skill.action_sequence
        self.compiled: Optional[Callable] = None
        self.usage_count = 0
        self.avg_duration_ms = 0.0

    def set_compiled(self, fn: Callable) -> None:
        self.compiled = fn

    def __call__(self, tool_manager: Any, **params: Any) -> Any:
        self.usage_count += 1
        start = time.time()

        if self.compiled:
            result = self.compiled(tool_manager, **params)
        else:
            result = {"output": f"{self.name} executed (uncompiled)", "error": None}

        duration = (time.time() - start) * 1000
        self.avg_duration_ms = (
            (self.avg_duration_ms * (self.usage_count - 1) + duration)
            / self.usage_count
        )
        return result


class MacroRegistry:
    min_usage_for_promotion: int = 10
    min_confidence_for_promotion: float = 0.85

    def __init__(self):
        self.macros: dict[str, SkillMacro] = {}

    def promote_to_macro(self, skill: _SkillStub) -> SkillMacro:
        macro = SkillMacro(skill)
        self.macros[skill.name] = macro
        logger.info(
            "Promoted skill '%s' to macro (%d uses, %.0f%% confidence)",
            skill.name, skill.usage_count, skill.confidence * 100,
        )
        return macro

    def try_execute(self, skill_name: str, tool_manager: Any, **params: Any) -> Optional[Any]:
        macro = self.macros.get(skill_name)
        if macro:
            return macro(tool_manager, **params)
        return None

    def clear(self) -> None:
        self.macros.clear()

    def size(self) -> int:
        return len(self.macros)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macros": {k: {"name": v.name, "usage_count": v.usage_count} for k, v in self.macros.items()}
        }


@dataclass
class _SkillResult:
    skill_name: str = ""
    steps_executed: int = 0
    compiled_steps: int = 0
    llm_steps: int = 0
    output: dict = field(default_factory=dict)
    total_cost: float = 0.0


class HybridSkillExecutor:
    def __init__(self):
        self.parser = SkillParser()
        self._compiled_steps: dict[str, Callable] = {}

    def register_compiled_step(self, step_name: str, fn: Callable) -> None:
        self._compiled_steps[step_name] = fn

    def execute(
        self,
        skill: _SkillStub,
        task: str,
        context: Optional[dict] = None,
    ) -> _SkillResult:
        ir = self.parser.parse(skill)
        context = context or {}
        results: dict[str, Any] = {}
        compiled_count = 0
        llm_count = 0

        for i, step in enumerate(ir.tools):
            if step in self._compiled_steps:
                results[step] = self._compiled_steps[step](context)
                compiled_count += 1
            else:
                results[step] = {"output": f"LLM would handle {step}", "error": None}
                llm_count += 1

        return _SkillResult(
            skill_name=skill.name,
            steps_executed=len(ir.tools),
            compiled_steps=compiled_count,
            llm_steps=llm_count,
            output=results,
        )


class SkillCompilationEngine:
    def __init__(self):
        self.parser = SkillParser()
        self.optimizer = SkillOptimizer()
        self.codegen = SkillCodeGenerator()
        self.executor = HybridSkillExecutor()
        self.macro_registry = MacroRegistry()
        self.learner = CompilationLearner()
        self._compiled_functions: dict[str, Callable] = {}

    def process_skill(self, skill: _SkillStub) -> CompilationResult:
        should, level = self.learner.should_compile(skill)

        if not should:
            return CompilationResult(
                skill_name=skill.name,
                level=CompilationLevel.TEXT_FRAGMENT,
                compiled=False,
            )

        ir = self.parser.parse(skill)
        ir = self.optimizer.optimize(ir)

        compiled: Any = None
        if level == CompilationLevel.NATIVE_CODE:
            code = self.codegen.generate_python(ir)
            compiled = self._exec_to_function(code)
            self._compiled_functions[skill.name] = compiled
        elif level == CompilationLevel.DSL_SCRIPT:
            compiled = self.codegen.generate_dsl(ir)
        elif level == CompilationLevel.CONSTRAINED:
            compiled = self.codegen.generate_constrained_grammar(ir)

        if skill.usage_count >= MacroRegistry.min_usage_for_promotion:
            macro = self.macro_registry.promote_to_macro(skill)
            if compiled and callable(compiled):
                macro.set_compiled(compiled)

        return CompilationResult(
            skill_name=skill.name,
            level=level,
            compiled=True,
            compiled_object=compiled,
            ir=ir,
        )

    def execute_compiled(self, skill_name: str, context: Optional[dict] = None) -> Any:
        result = self.macro_registry.try_execute(skill_name, _ToolManagerStub(), **(context or {}))
        if result is not None:
            return result

        fn = self._compiled_functions.get(skill_name)
        if fn:
            return fn(_ToolManagerStub(), **(context or {}))

        raise ValueError(f"Skill '{skill_name}' not compiled")

    def _exec_to_function(self, code: str) -> Callable:
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        key = [k for k in namespace if k.startswith("compiled_")]
        if key:
            return namespace[key[0]]
        return lambda *a, **kw: {"output": None, "error": "Compilation failed"}
