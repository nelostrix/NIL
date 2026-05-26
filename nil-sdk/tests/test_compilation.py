from __future__ import annotations

import pytest

from nil.compilation.pipeline import (
    CompilationLevel,
    ControlFlowNode,
    IntermediateRepresentation,
    _SkillStub,
    SkillParser,
    SkillOptimizer,
    SkillCodeGenerator,
    CompiledSkillExecutor,
    CompilationLearner,
    HybridSkillExecutor,
    SkillCompilationEngine,
    SkillMacro,
    MacroRegistry,
    CompilationResult,
)


class TestCompilationLevel:
    def test_values(self):
        assert CompilationLevel.TEXT_FRAGMENT.value == 0
        assert CompilationLevel.NATIVE_CODE.value == 4


class TestIntermediateRepresentation:
    def test_defaults(self):
        ir = IntermediateRepresentation(name="test", tools=["a", "b"])
        assert ir.is_linear is True
        assert ir.estimated_steps == 2
        assert ir.can_compile is True

    def test_non_linear(self):
        ir = IntermediateRepresentation(
            name="test", tools=["a", "b"],
            has_conditionals=True,
            control_flow=[ControlFlowNode(node_type="condition")],
        )
        assert ir.is_linear is False
        assert ir.can_compile is False

    def test_to_dict_roundtrip(self):
        ir = IntermediateRepresentation(name="test", tools=["a", "b"], confidence=0.9)
        d = ir.to_dict()
        restored = IntermediateRepresentation.from_dict(d)
        assert restored.name == "test"
        assert restored.tools == ["a", "b"]
        assert restored.confidence == 0.9


class TestControlFlowNode:
    def test_defaults(self):
        n = ControlFlowNode(node_type="tool_call", tool_name="search")
        assert n.node_type == "tool_call"
        assert n.tool_name == "search"


class TestSkillParser:
    def test_parse_with_action_sequence(self):
        parser = SkillParser()
        skill = _SkillStub(
            name="investigate",
            action_sequence=["search_logs", "analyze_logs"],
            description="Search logs then analyze",
        )
        ir = parser.parse(skill)
        assert ir.name == "investigate"
        assert ir.tools == ["search_logs", "analyze_logs"]
        assert ir.source == "parsed"

    def test_parse_from_description(self):
        parser = SkillParser()
        skill = _SkillStub(
            name="test",
            description="First search_logs for errors, then analyze_logs",
        )
        ir = parser.parse(skill)
        assert len(ir.tools) >= 1

    def test_parse_empty(self):
        parser = SkillParser()
        skill = _SkillStub(name="empty", description="do something unknown")
        ir = parser.parse(skill)
        assert ir.tools == []
        assert ir.source == "text_only"

    def test_parse_conditional(self):
        parser = SkillParser()
        skill = _SkillStub(name="conditional", description="if error occurs then retry")
        ir = parser.parse(skill)
        assert ir.source == "text_only"


class TestSkillOptimizer:
    def test_linear_optimization(self):
        opt = SkillOptimizer()
        ir = IntermediateRepresentation(name="test", tools=["a", "b", "c"])
        result = opt.optimize(ir)
        assert len(result.tools) >= 1

    def test_fusion(self):
        opt = SkillOptimizer()
        opt.register_fusion("search_logs", "analyze_logs", "search_and_analyze")
        ir = IntermediateRepresentation(name="test", tools=["search_logs", "analyze_logs", "c"])
        result = opt.optimize(ir)
        assert "search_and_analyze" in result.tools

    def test_specialization(self):
        opt = SkillOptimizer()
        opt.register_specialization("search_logs", "search_logs_with_timeout")
        ir = IntermediateRepresentation(name="test", tools=["search_logs"])
        result = opt.optimize(ir)
        assert result.tools == ["search_logs_with_timeout"]

    def test_dead_step_elimination(self):
        opt = SkillOptimizer()
        ir = IntermediateRepresentation(name="test", tools=["a", "b", "c"])
        result = opt._eliminate_dead_steps(ir)
        assert len(result.tools) >= 2

    def test_fuse_adjacent_pairs(self):
        opt = SkillOptimizer()
        ir = IntermediateRepresentation(name="test", tools=["a", "b", "c"])
        result = opt._fuse_adjacent_pairs(ir)
        assert result.tools == ["a", "b", "c"]


class TestSkillCodeGenerator:
    def test_generate_python(self):
        gen = SkillCodeGenerator()
        ir = IntermediateRepresentation(name="test", tools=["search_logs", "analyze_logs"])
        code = gen.generate_python(ir)
        assert "def compiled_test" in code
        assert "search_logs" in code
        assert "analyze_logs" in code

    def test_generate_dsl(self):
        gen = SkillCodeGenerator()
        ir = IntermediateRepresentation(
            name="test", tools=["search_logs"],
            preconditions=["service is up"],
            postconditions=["results found"],
        )
        dsl = gen.generate_dsl(ir)
        assert "SKILL test" in dsl
        assert "search_logs" in dsl
        assert "PRECONDITIONS" in dsl
        assert "POSTCONDITIONS" in dsl

    def test_generate_constrained(self):
        gen = SkillCodeGenerator()
        gen.register_tool_info("search_logs", "Search tool", {"q": {"type": "string"}})
        ir = IntermediateRepresentation(name="test", tools=["search_logs"])
        grammar = gen.generate_constrained_grammar(ir)
        assert grammar["skill_name"] == "test"
        assert len(grammar["steps"]) == 1


class TestCompiledSkillExecutor:
    def test_compile_and_cache_native(self):
        executor = CompiledSkillExecutor()
        skill = _SkillStub(
            name="test_skill",
            action_sequence=["search_logs", "analyze_logs"],
        )
        result = executor.compile_and_cache(skill, CompilationLevel.NATIVE_CODE)
        assert callable(result)

    def test_compile_and_cache_dsl(self):
        executor = CompiledSkillExecutor()
        skill = _SkillStub(name="test", action_sequence=["search_logs"])
        result = executor.compile_and_cache(skill, CompilationLevel.DSL_SCRIPT)
        assert isinstance(result, str)
        assert "SKILL test" in result

    def test_compile_and_cache_constrained(self):
        executor = CompiledSkillExecutor()
        skill = _SkillStub(name="test", action_sequence=["search_logs"])
        result = executor.compile_and_cache(skill, CompilationLevel.CONSTRAINED)
        assert isinstance(result, dict)

    def test_execute_compiled(self):
        executor = CompiledSkillExecutor()
        skill = _SkillStub(name="exec_test", action_sequence=["search_logs"])
        executor.compile_and_cache(skill, CompilationLevel.NATIVE_CODE)
        result = executor.execute("exec_test")
        assert result is not None

    def test_execute_not_compiled(self):
        executor = CompiledSkillExecutor()
        with pytest.raises(ValueError, match="not compiled"):
            executor.execute("nonexistent")


class TestCompilationLearner:
    def test_below_min_uses(self):
        learner = CompilationLearner()
        skill = _SkillStub(name="test", usage_count=1, confidence=0.9)
        should, level = learner.should_compile(skill)
        assert should is False

    def test_native_code_level(self):
        learner = CompilationLearner()
        learner.add_trace_for_skill("test", ["a", "b"])
        learner.add_trace_for_skill("test", ["a", "b"])
        skill = _SkillStub(name="test", usage_count=5, confidence=0.95)
        should, level = learner.should_compile(skill)
        assert should is True
        assert level == CompilationLevel.NATIVE_CODE

    def test_dsl_level(self):
        learner = CompilationLearner()
        learner.add_trace_for_skill("test", ["a", "b"])
        learner.add_trace_for_skill("test", ["a", "b"])
        skill = _SkillStub(name="test", usage_count=5, confidence=0.85)
        should, level = learner.should_compile(skill)
        assert should is True
        assert level == CompilationLevel.DSL_SCRIPT

    def test_low_determinism_keeps_text(self):
        learner = CompilationLearner()
        skill = _SkillStub(name="test", usage_count=5, confidence=0.5)
        should, level = learner.should_compile(skill)
        assert should is False
        assert level == CompilationLevel.TEXT_FRAGMENT


class TestSkillMacro:
    def test_call(self):
        skill = _SkillStub(name="test", action_sequence=["search_logs"])
        macro = SkillMacro(skill)
        result = macro(_ToolManagerStub())
        assert result is not None
        assert macro.usage_count == 1

    def test_call_with_compiled(self):
        skill = _SkillStub(name="test", action_sequence=["search_logs"])
        macro = SkillMacro(skill)
        def fake_fn(tm, **kw):
            return {"output": "compiled", "error": None}
        macro.set_compiled(fake_fn)
        result = macro(_ToolManagerStub())
        assert result["output"] == "compiled"


class _ToolManagerStub:
    def execute(self, tool_name, kwargs):
        return {"output": f"{tool_name} executed", "error": None}


class TestMacroRegistry:
    def test_promote_and_execute(self):
        registry = MacroRegistry()
        skill = _SkillStub(name="test_macro", usage_count=15, confidence=0.9)
        macro = registry.promote_to_macro(skill)
        assert macro.name == "test_macro"
        assert registry.size() == 1

    def test_try_execute_nonexistent(self):
        registry = MacroRegistry()
        result = registry.try_execute("nonexistent", _ToolManagerStub())
        assert result is None

    def test_clear(self):
        registry = MacroRegistry()
        registry.promote_to_macro(_SkillStub(name="s1"))
        registry.clear()
        assert registry.size() == 0


class TestHybridSkillExecutor:
    def test_execute(self):
        executor = HybridSkillExecutor()
        skill = _SkillStub(
            name="test",
            action_sequence=["search_logs", "analyze_logs"],
        )
        result = executor.execute(skill, "investigate latency")
        assert result.skill_name == "test"
        assert result.steps_executed == 2

    def test_mixed_compiled_and_llm(self):
        executor = HybridSkillExecutor()
        executor.register_compiled_step("search_logs", lambda ctx: {"output": "found", "error": None})
        skill = _SkillStub(name="test", action_sequence=["search_logs", "analyze_logs"])
        result = executor.execute(skill, "test")
        assert result.compiled_steps == 1
        assert result.llm_steps == 1


class TestSkillCompilationEngine:
    def test_process_uncompilable(self):
        engine = SkillCompilationEngine()
        skill = _SkillStub(name="new_skill", usage_count=1, confidence=0.5)
        result = engine.process_skill(skill)
        assert result.compiled is False
        assert result.level == CompilationLevel.TEXT_FRAGMENT

    def test_process_compilable(self):
        engine = SkillCompilationEngine()
        engine.learner.add_trace_for_skill("stable_skill", ["a", "b"])
        engine.learner.add_trace_for_skill("stable_skill", ["a", "b"])
        skill = _SkillStub(
            name="stable_skill",
            action_sequence=["search_logs", "analyze_logs"],
            usage_count=15,
            confidence=0.95,
        )
        result = engine.process_skill(skill)
        assert result.compiled is True

    def test_execute_compiled(self):
        engine = SkillCompilationEngine()
        engine.learner.add_trace_for_skill("exec_skill", ["a", "b"])
        engine.learner.add_trace_for_skill("exec_skill", ["a", "b"])
        skill = _SkillStub(
            name="exec_skill",
            action_sequence=["search_logs"],
            usage_count=15,
            confidence=0.95,
        )
        engine.process_skill(skill)
        result = engine.execute_compiled("exec_skill")
        assert result is not None

    def test_execute_not_compiled_raises(self):
        engine = SkillCompilationEngine()
        with pytest.raises(ValueError):
            engine.execute_compiled("nonexistent")


class TestCompilationResult:
    def test_to_dict(self):
        ir = IntermediateRepresentation(name="test", tools=["a"])
        result = CompilationResult(
            skill_name="test",
            level=CompilationLevel.NATIVE_CODE,
            compiled=True,
            ir=ir,
        )
        d = result.to_dict()
        assert d["skill_name"] == "test"
        assert d["level"] == 4
        assert d["compiled"] is True
        assert d["ir"]["name"] == "test"
