# Skill Compilation

> **Status: Partially Implemented** — The compilation pipeline exists in `nil/compilation/pipeline.py`. `SkillParser` (regex-based tool extraction), `IntermediateRepresentation`, `SkillOptimizer` (fusion/specialization), `SkillCodeGenerator` (linear Python output), `HybridSkillExecutor`, `CompilationLearner`, `SkillMacro`, `MacroRegistry`, and `SkillCompilationEngine` are all implemented. Compilation up to `COMPILED_TREE` level is supported. **Not yet implemented:** `DSL_SCRIPT` and `NATIVE_BINARY` compilation levels, conditional/loop control flow, cross-skill composition, and template instantiation.

---

## Background

Skills in Nil were originally stored as text — a name, description, compressed instruction, and tool sequence. They were injected into prompts as natural language fragments. The agent read the text and (hopefully) followed the instruction. This worked but was fragile: there was no guarantee the LLM would execute the skill correctly, no notion of a skill as an executable program, no compilation into deterministic behavior.

*Skill compilation* converts learned textual skills into executable policies, deterministic workflows, reusable planners, and parameterized macros. The goal: skills that are not just *read* by the LLM but *executed* by the runtime.

---

## Before Compilation

```
Textual Skill (current):

  SKILL: search_logs_then_analyze
    WHEN: Task involves investigating latency or errors
    DO: First, search logs, then analyze the output
    CONFIDENCE: 92% (14 uses)

  What happens at runtime:
    The LLM reads this text and may or may not follow it.
    There is no enforcement, no execution, no guarantees.

  Limitations:
    • Non-deterministic (LLM may ignore the skill)
    • No parameterization (skill is a fixed text, not a callable)
    • No composition (skills cannot call other skills)
    • No compilation (no way to verify correctness ahead of time)
    • No optimization (skill is text, cannot be optimized numerically)
    • No abstraction (skill mixes "what" with "how")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SKILL COMPILATION PIPELINE                   │
│                                                                     │
│  Text Skill ──→ Parse ──→ IR ──→ Optimize ──→ Code Gen ──→ Exec    │
│                                                                     │
│  Stages:                                                             │
│    1. Parse: Extract structure from natural language skill           │
│    2. IR: Intermediate representation (DAG of atomic operations)     │
│    3. Optimize: Dead-step elimination, fusion, constant folding      │
│    4. Code Gen: Emit Python/DSL/constrained generation               │
│    5. Exec: Run compiled skill deterministically                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Skill Representation Spectrum

Skills exist on a spectrum from fully textual to fully compiled:

```
Textual ───────────────────────────────────────────────→ Compiled

  Prompt        Few-shot       Constrained      DSL         Native
  Fragment      Examples       Generation       Program     Code

  "Search      [Example 1]    "Always call     def inves-   def inves-
  logs then    [Example 2]    search_logs      tigate():   tigate():
  analyze"                    before call-        search()   result =
                                ing analyze"      analyze()  search(...)
                                                           return analyze(
                                                             result)

  Flexibility decreases ────────────────────────────→ Determinism increases
  LLM required  ────────────────────────────────────→ No LLM needed
```

### Compilation Levels

```python
class CompilationLevel(Enum):
    TEXT_FRAGMENT   = 0   # Raw text injected into prompt
    FEW_SHOT        = 1   # Example traces included in context
    CONSTRAINED     = 2   # Hard constraints in system prompt
    DSL_SCRIPT      = 3   # Executable DSL program
    NATIVE_CODE     = 4   # Compiled Python function
    COMPILED_TREE   = 5   # Decision tree / policy (no LLM)
```

---

## 2. Compilation Pipeline

### Stage 1: Parse

```python
class SkillParser:
    """
    Parse a textual skill into structured intermediate representation.

    Input:  "First, search logs for error patterns, then analyze
              the results to identify root cause."

    Output: IR (list of atomic operations with control flow)
    """

    def parse(self, skill: Skill) -> IntermediateRepresentation:
        """
        Extract structure from natural language skill description.

        Methods:
          - Regex patterns for common skill structures
          - LLM-assisted parsing for complex skills
          - Graph extraction from tool sequences
        """
        # Extract tool sequence
        tools = self._extract_tool_sequence(skill)

        # Extract control flow (sequential / conditional / parallel)
        control_flow = self._extract_control_flow(skill.description)

        # Extract preconditions and postconditions
        pre, post = skill.preconditions, skill.postconditions

        return IntermediateRepresentation(
            name=skill.name,
            tools=tools,
            control_flow=control_flow,
            preconditions=pre,
            postconditions=post,
            confidence=skill.confidence,
            source="parsed" if tools else "text_only",
        )

    def _extract_tool_sequence(self, skill: Skill) -> list[str]:
        """Extract tool sequence from skill's action_sequence or description."""
        if skill.action_sequence:
            return skill.action_sequence

        # Fall back: extract tool names from description text
        return self._find_tool_names(skill.description)
```

### Stage 2: Intermediate Representation

```python
@dataclass
class IntermediateRepresentation:
    """
    Structured, machine-readable representation of a skill.

    This is the IR that optimization and code generation operate on.
    """
    name: str
    tools: list[str]
    control_flow: list[ControlFlowNode]
    preconditions: list[str]
    postconditions: list[str]
    confidence: float
    source: str

    # Derived
    is_linear: bool              # True if no branching
    has_conditionals: bool
    estimated_steps: int
    can_compile: bool            # True if all tools are known + deterministic


@dataclass
class ControlFlowNode:
    node_type: Literal["tool_call", "condition", "loop", "parallel", "subskill"]
    tool_name: Optional[str] = None
    condition: Optional[str] = None
    children: list['ControlFlowNode'] = None
    subskill_name: Optional[str] = None
```

### Stage 3: Optimization

```python
class SkillOptimizer:
    """
    Optimizes skill IR before code generation.

    Transforms:
      - Dead step elimination (remove tools that don't affect outcome)
      - Fusion (merge adjacent tool calls that are always paired)
      - Constant folding (pre-compute static parameters)
      - Specialization (specialize generic skills for specific domains)
      - Compaction (remove redundant verification steps)
    """

    def optimize(self, ir: IntermediateRepresentation) -> IntermediateRepresentation:
        if ir.is_linear:
            ir = self._eliminate_dead_steps(ir)
            ir = self._fuse_adjacent_pairs(ir)
        ir = self._specialize(ir)
        return ir

    def _eliminate_dead_steps(self, ir: IR) -> IR:
        """
        Remove steps whose output is never consumed.

        If tool A's output is never passed to tool B or the final
        result, tool A is dead and can be eliminated.
        """
        if len(ir.tools) < 2:
            return ir

        # Trace data dependencies between consecutive tools
        alive = [ir.tools[0]]  # first tool always needed
        for i in range(1, len(ir.tools)):
            prev = ir.tools[i - 1]
            curr = ir.tools[i]
            # If prev output is used by curr, keep prev
            # Otherwise, prev was a dead end
            if self._output_consumed_by(prev, curr):
                alive.append(prev)
        alive.append(ir.tools[-1])

        # Deduplicate consecutive
        unique = [alive[0]]
        for t in alive[1:]:
            if t != unique[-1]:
                unique.append(t)

        ir.tools = unique
        return ir

    def _fuse_adjacent_pairs(self, ir: IR) -> IR:
        """
        Merge tool pairs that always appear together.

        If search_logs is always followed by analyze_logs,
        they become a fused unit: search_and_analyze.

        This reduces steps and lets the composite be compiled
        as a single call.
        """
        fused = []
        i = 0
        while i < len(ir.tools):
            if i + 1 < len(ir.tools):
                pair = (ir.tools[i], ir.tools[i + 1])
                if pair in self.fusion_map:  # known fused pair
                    fused.append(self.fusion_map[pair])
                    i += 2
                    continue
            fused.append(ir.tools[i])
            i += 1
        ir.tools = fused
        return ir

    def _specialize(self, ir: IR) -> IR:
        """
        Specialize generic tools for the specific domain.

        Generic:  "search_logs(query)"
        Specialized: "search_logs_with_timeout(query, timerange='last_30m')"

        Specialization bakes in domain knowledge as default parameters.
        """
        specialized_tools = []
        for tool_name in ir.tools:
            if tool_name in self.specialization_map:
                specialized_tools.append(self.specialization_map[tool_name])
            else:
                specialized_tools.append(tool_name)
        ir.tools = specialized_tools
        return ir
```

### Stage 4: Code Generation

```python
class SkillCodeGenerator:
    """
    Generates executable code from optimized skill IR.

    Output formats:
      - Python function (deterministic, no LLM needed)
      - Constrained generation grammar (for LLM-guided execution)
      - DSL script (interpreted by skill runtime)
    """

    def generate_python(self, ir: IntermediateRepresentation) -> str:
        """
        Generate a deterministic Python function.

        The compiled skill runs without LLM involvement —
        purely tool calls with data flow between them.
        """
        lines = [
            f"def compiled_{ir.name}(tool_manager, **kwargs):",
            f"    \"\"\"Compiled skill: {ir.name}\"\"\"",
            "    results = {}",
            "",
        ]

        for i, tool_name in enumerate(ir.tools):
            var_name = f"result_{i}"
            lines.extend([
                f"    # Step {i + 1}: {tool_name}",
                f"    {var_name} = tool_manager.execute(",
                f"        '{tool_name}', kwargs",
                f"    )",
                f"    if {var_name}['error']:",
                f"        return {{'error': {var_name}['error'], 'step': {i}}}",
                f"    results['{tool_name}'] = {var_name}['output']",
                "",
            ])

        lines.append("    return {'output': results, 'error': None}")
        return "\n".join(lines)

    def generate_constrained_grammar(self, ir: IR) -> dict:
        """
        Generate a constrained generation grammar for LLM-guided execution.

        Instead of "think what to do", the LLM fills in parameters
        for a pre-determined tool sequence.
        """
        steps = []
        for tool_name in ir.tools:
            tool = self.tool_manager.get(tool_name)
            steps.append({
                "step_type": "tool_call",
                "tool_name": tool_name,
                "description": tool.description,
                "parameters": {
                    k: {"type": v.get("type", "string"), "description": v.get("description", "")}
                    for k, v in tool.parameters.items()
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
        """
        Generate a DSL script for the skill runtime.

        DSL is an intermediate format — more structured than text,
        more flexible than native code.
        """
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
```

### Stage 5: Execution

```python
class CompiledSkillExecutor:
    """
    Executes compiled skills.

    Supports four execution modes:
      - LLM-guided (LLM follows constrained grammar)
      - DSL-interpreted (skill runtime interprets DSL)
      - Native (calls compiled Python function)
      - Hybrid (LLM for complex steps, native for deterministic)
    """

    def __init__(self):
        self._compiled_cache: dict[str, Callable] = {}

    def compile_and_cache(self, skill: Skill, level: CompilationLevel):
        """Compile a skill and cache the result."""
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

    def execute(self, skill_name: str, context: dict) -> Any:
        """Execute a compiled skill by name."""
        compiled = self._compiled_cache.get(skill_name)
        if compiled:
            return compiled(self.tool_manager, **context)
        raise SkillNotCompiledError(skill_name)
```

---

## 3. Skill Forms by Compilation Level

### Level 0: Text Fragment (current)

```
Injected as: "SKILL: search_logs_then_analyze: When investigating, search logs then analyze."
Execution: LLM reads it and may or may not comply.
```

### Level 1: Few-Shot Example

```
Injected as an example trace:
  Task: "Investigate API latency"
  Steps:
    1. search_logs("query=latency", "timerange=30m") → results
    2. analyze_logs(results) → root cause
  Outcome: success

Execution: LLM has a concrete example to pattern-match.
```

### Level 2: Constrained Generation

```python
# Injected as a structured block:
{
  "skill": "investigate_latency",
  "sequence": [
    {"tool": "search_logs", "params": {"query": "<fill>", "timerange": "30m"}},
    {"tool": "analyze_logs", "params": {"data": "from_step_1"}},
  ],
  "rules": [
    "Always call search_logs before analyze_logs",
    "If search_logs returns empty, try query_metrics instead",
  ]
}

# LLM fills in parameters within structure.
# Tool order is enforced. No deviation from the skill.
```

### Level 3: DSL Script

```
SKILL investigate_latency
  PRECONDITIONS: service is accessible, logs are being written
  STEP 1: CALL search_logs(query=$task_description, timerange="30m")
    ON_ERROR: fallback query_metrics
    OUTPUT: $logs
  STEP 2: CALL analyze_logs(data=$logs)
    ON_ERROR: retry with expanded timerange
    OUTPUT: $analysis
  STEP 3: CALL diagnose(data=$logs, $analysis)
    OUTPUT: $diagnosis
  POSTCONDITIONS: root cause identified
  OUTPUT: $diagnosis
```

### Level 4: Native Python

```python
def compiled_investigate_latency(tool_manager, task_description, **kwargs):
    # Step 1: Search logs
    logs = tool_manager.execute("search_logs", {
        "query": task_description,
        "timerange": "30m",
    })
    if logs["error"]:
        logs = tool_manager.execute("query_metrics", {
            "query": task_description,
        })

    # Step 2: Analyze
    analysis = tool_manager.execute("analyze_logs", {
        "data": logs["output"],
    })

    # Step 3: Diagnose
    diagnosis = tool_manager.execute("diagnose", {
        "logs": logs["output"],
        "analysis": analysis["output"],
    })

    return diagnosis
```

### Level 5: Compiled Decision Tree

```python
class CompiledSkillTree:
    """
    A skill compiled into a decision tree — no LLM required at all.

    Each node is a tool call or condition. Edges are outcomes.
    The tree is learned from traces and optimized for accuracy.
    """

    def execute(self, state: EnvironmentState, tool_manager) -> Action:
        node = self.root
        while not node.is_leaf:
            if node.is_decision:
                # Branch based on state
                node = node.branches[self._evaluate_condition(node.condition, state)]
            else:
                # Execute tool, use result to decide next
                result = tool_manager.execute(node.tool_name)
                node = node.branches[self._classify_result(result)]
        return node.action
```

---

## 4. Macro System

Frequent skill sequences become macros — single-call abstractions:

```python
class SkillMacro:
    """
    A frequently used skill sequence registered as a single callable.

    Once a skill reaches sufficient confidence and frequency,
    it becomes a macro — callable by name with no LLM involvement
    for the internal steps.
    """

    def __init__(self, skill: Skill):
        self.name = skill.name
        self.steps = skill.action_sequence
        self.compiled = self._compile(skill)
        self.usage_count = 0
        self.avg_duration_ms = 0.0

    def __call__(self, tool_manager, **params) -> Any:
        self.usage_count += 1
        start = time.time()

        if self.compiled:
            result = self.compiled(tool_manager, **params)
        else:
            # Fall back to LLM-guided execution
            result = self._llm_guided_execute(tool_manager, params)

        duration = (time.time() - start) * 1000
        self.avg_duration_ms = (
            (self.avg_duration_ms * (self.usage_count - 1) + duration)
            / self.usage_count
        )

        return result


class MacroRegistry:
    """
    Registry of compiled skill macros.

    Macros are upgraded from candidate → active based on
    frequency and confidence thresholds.
    """

    min_usage_for_promotion: int = 10
    min_confidence_for_promotion: float = 0.85

    def promote_to_macro(self, skill: Skill) -> SkillMacro:
        """Promote a skill to a callable macro."""
        macro = SkillMacro(skill)
        self.macros[skill.name] = macro
        logger.info(
            "Promoted skill '%s' to macro (%d uses, %.0f%% confidence)",
            skill.name, skill.usage_count, skill.confidence * 100,
        )
        return macro

    def try_execute(self, skill_name: str, tool_manager, **params) -> Optional[Any]:
        """Execute a macro by name if it exists."""
        macro = self.macros.get(skill_name)
        if macro:
            return macro(tool_manager, **params)
        return None
```

---

## 5. Learning to Compile

The compilation target is itself learned:

```python
class CompilationLearner:
    """
    Learns when and how to compile skills.

    Decides:
      - Which skills to compile (confidence × frequency × determinism)
      - What compilation level to target
      - When to upgrade compilation level
      - When to revert (if compiled version underperforms text version)
    """

    def should_compile(self, skill: Skill) -> tuple[bool, CompilationLevel]:
        """Determine if a skill should be compiled and to what level."""
        if skill.usage_count < self.config.min_uses_for_compilation:
            return False, CompilationLevel.TEXT_FRAGMENT

        # Measure determinism of the skill's execution
        determinism = self._measure_determinism(skill)

        if determinism > 0.95 and skill.confidence > 0.9:
            # Highly deterministic → native code
            return True, CompilationLevel.NATIVE_CODE
        elif determinism > 0.8 and skill.confidence > 0.8:
            # Mostly deterministic → DSL
            return True, CompilationLevel.DSL_SCRIPT
        elif determinism > 0.6:
            # Somewhat deterministic → constrained generation
            return True, CompilationLevel.CONSTRAINED
        else:
            # Not deterministic enough → keep as text
            return False, CompilationLevel.TEXT_FRAGMENT

    def _measure_determinism(self, skill: Skill) -> float:
        """
        Measure how deterministically the skill executes.

        A skill is deterministic if:
          - Same input always produces same tool sequence
          - Same input always produces same output
          - No branching based on LLM choice

        Measured from trace data.
        """
        traces = self.memory.get_traces_for_skill(skill.name)
        if len(traces) < 5:
            return 0.0

        # Check if tool sequence is always the same
        sequences = [t.tool_sequence for t in traces]
        first_seq = sequences[0]
        matches = sum(1 for s in sequences if s == first_seq)
        sequence_consistency = matches / len(sequences)

        # Check if parameter values are consistent
        param_consistency = self._parameter_consistency(traces)

        return 0.6 * sequence_consistency + 0.4 * param_consistency
```

---

## 6. Hybrid Execution

Not all skills need to be fully compiled. Hybrid execution mixes LLM reasoning with compiled determinism:

```python
class HybridSkillExecutor:
    """
    Executes skills using the best available compilation level.

    For each step in a skill:
      - If step is compiled: run deterministically (fast, cheap, reliable)
      - If step needs reasoning: route to LLM (slow, expensive, flexible)

    This gives the best of both worlds.
    """

    def execute(
        self,
        skill: Skill,
        task: str,
        context: dict,
    ) -> SkillResult:
        ir = self.parser.parse(skill)

        results = {}
        for i, step in enumerate(ir.tools):
            if self._is_compiled(step):
                # Deterministic execution
                results[step] = self._exec_compiled(step, context)
            else:
                # LLM-guided execution with constraint
                results[step] = self._exec_with_llm(
                    step, results, task, context
                )

        return SkillResult(
            skill_name=skill.name,
            steps_executed=len(ir.tools),
            compiled_steps=sum(1 for t in ir.tools if self._is_compiled(t)),
            llm_steps=sum(1 for t in ir.tools if not self._is_compiled(t)),
            output=results,
            total_cost=sum(r.cost for r in results.values()),
        )
```

---

## 7. Integration with Nil

```python
class SkillCompilationEngine:
    """
    Full compilation pipeline integrated into the agent.

    Runs as part of the evolution cycle:
      After skill extraction → try to compile
      After run completion → update compilation statistics
      Periodically → upgrade compilation levels
    """

    def __init__(self, memory: PersistentMemory, tool_manager: ToolManager):
        self.parser = SkillParser()
        self.optimizer = SkillOptimizer()
        self.codegen = SkillCodeGenerator(tool_manager)
        self.executor = HybridSkillExecutor()
        self.macro_registry = MacroRegistry()
        self.learner = CompilationLearner(memory)
        self.tool_manager = tool_manager

    def process_skill(self, skill: Skill) -> CompilationResult:
        """Process a single skill through the compilation pipeline."""
        should, level = self.learner.should_compile(skill)

        if not should:
            return CompilationResult(
                skill_name=skill.name,
                level=CompilationLevel.TEXT_FRAGMENT,
                compiled=False,
            )

        # Compile
        ir = self.parser.parse(skill)
        ir = self.optimizer.optimize(ir)

        if level == CompilationLevel.NATIVE_CODE:
            code = self.codegen.generate_python(ir)
            compiled = self._exec_to_function(code)

        elif level == CompilationLevel.DSL_SCRIPT:
            compiled = self.codegen.generate_dsl(ir)

        elif level == CompilationLevel.CONSTRAINED:
            compiled = self.codegen.generate_constrained_grammar(ir)

        # Promote to macro if applicable
        if skill.usage_count >= MacroRegistry.min_usage_for_promotion:
            self.macro_registry.promote_to_macro(skill)

        return CompilationResult(
            skill_name=skill.name,
            level=level,
            compiled=True,
            compiled_object=compiled,
            ir=ir,
        )

    def execute_compiled(self, skill_name: str, context: dict) -> Any:
        """Execute a compiled skill."""
        # Try macro first (fastest)
        result = self.macro_registry.try_execute(
            skill_name, self.tool_manager, **context
        )
        if result is not None:
            return result

        # Fall back to hybrid executor
        skill = self.memory.get_skill(context["domain"], skill_name)
        return self.executor.execute(skill, context["task"], context)
```

---

## Implementation Status

| Phase | What | Status |
|---|---|---|
| 1 | Skill parser | **Done** — regex-based tool sequence extraction from skill descriptions |
| 2 | IR format | **Done** — `IntermediateRepresentation` dataclass with linear/conditional detection, optimization passes |
| 3 | Dead-step elimination | **Done** — data-dependency tracing, removes unconsumed outputs |
| 4 | Tool fusion | **Done** — fusion map merging always-adjacent pairs |
| 5 | Constrained generation grammar | **Done** — structured tool sequence with parameter fill-in for LLM |
| 6 | DSL interpreter | **Not implemented** — no DSL runtime yet |
| 7 | Native code generation | **Done** — Python function generation + `exec` to callable |
| 8 | Compilation learner | **Done** — determinism measurement, level selection (confidence × frequency) |
| 9 | Macro system | **Done** — `SkillMacro` / `MacroRegistry` with promotion from candidate → active |
| 10 | Hybrid executor | **Done** — mixes compiled and LLM-guided steps within one skill |

---

## Key Differentiator

**Textual skills only:**
The agent reads instructions and hopes to follow them. Every execution is a fresh act of interpretation. There is no guarantee, no optimization, no compilation. The skill is advice, not capability.

**With skill compilation:**
Skills become executable assets. The system determines which skills are deterministic enough to compile, generates optimized code, and registers them as callable macros. The agent executes them with the same reliability as a function call — no interpretation, no deviation, no cost per step.
