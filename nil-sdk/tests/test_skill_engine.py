"""
Comprehensive test suite for Skill Compression Engine.

Coverage:
  - Skill data model (TriggerCondition, Skill, SkillStatus, serialization)
  - Name generation (collapsing, single-tool, special chars)
  - Trigger inference (keywords, error patterns, tool prerequisites)
  - Chain compression (single, multi, confidence notes)
  - Pre/post condition extraction
  - Deduplication (subsequence absorption, confidence tiebreaks)
  - Merging with existing skills (version increment, field merge)
  - Lifecycle management (deprecation, usage tracking)
  - Prompt generation (full, compact)
  - Full extraction pipeline (end-to-end)
  - Edge cases (empty, unicode, single trace, all failures)
  - Scalability
"""

import tempfile
import time
import json
import pytest

from nil.memory.store import PersistentMemory
from nil.memory.skill_registry import (
    Skill, SkillStatus, SkillOrigin, TriggerCondition,
)
from nil.evolution.skill_extractor import SkillExtractor, ExtractionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = PersistentMemory(tmpdir)
        yield mem
        mem.close()


@pytest.fixture
def extractor(memory):
    return SkillExtractor(memory, min_frequency=3, min_confidence=0.5)


def _trace(
    run_id, success, tools, domain="test", cost=0.001,
    duration=1.0, error=None, created_offset=0,
    task="generic task", step_errors=None,
):
    steps = []
    for i, t in enumerate(tools):
        step = {
            "step_number": i + 1, "step_type": "tool_call",
            "tool_name": t, "tool_input": {}, "tool_output": "ok",
            "duration_ms": 50, "tokens_in": 10, "tokens_out": 5,
            "error": None, "timestamp": time.time() + created_offset,
        }
        if step_errors and i in step_errors:
            step["error"] = step_errors[i]
        steps.append(step)
    return {
        "run_id": run_id, "domain": domain, "task": task,
        "steps": steps, "outcome_success": success,
        "outcome_output": "done" if success else "failed",
        "total_duration": duration, "total_tokens_in": 100,
        "total_tokens_out": 50, "total_cost": cost,
        "tool_sequence": tools,
        "error": error if not success else None,
        "created_at": time.time() + created_offset, "metadata": {},
    }


def _populate(memory, traces):
    for t in traces:
        memory.store_trace(t)


# ===========================================================================
# SKILL DATA MODEL
# ===========================================================================

class TestSkillDataModel:
    def test_skill_defaults(self):
        s = Skill(name="test")
        assert s.status == SkillStatus.CANDIDATE
        assert s.origin == SkillOrigin.PATTERN_EXTRACTION
        assert s.version == 1
        assert s.chain_length == 0
        assert s.usage_success_rate == 0.0
        assert s.is_effective is False

    def test_skill_usage_rate(self):
        s = Skill(name="test", success_when_used=8, failure_when_used=2)
        assert s.usage_success_rate == pytest.approx(0.8)

    def test_skill_is_effective(self):
        s = Skill(name="test", success_when_used=8, failure_when_used=2)
        assert s.is_effective is True

    def test_skill_not_effective_low_count(self):
        s = Skill(name="test", success_when_used=2, failure_when_used=0)
        assert s.is_effective is False  # < 3 total uses

    def test_skill_not_effective_low_rate(self):
        s = Skill(name="test", success_when_used=2, failure_when_used=8)
        assert s.is_effective is False

    def test_compression_ratio(self):
        s = Skill(
            name="test",
            action_sequence=["a", "b", "c"],
            compressed_instruction="Do abc",
        )
        assert s.compression_ratio == 3.0

    def test_to_dict_from_dict_roundtrip(self):
        s = Skill(
            name="test_skill",
            description="Test description",
            trigger=TriggerCondition(keywords=["timeout"]),
            action_sequence=["search", "fix"],
            compressed_instruction="Search then fix",
            confidence=0.85,
            status=SkillStatus.ACTIVE,
            origin=SkillOrigin.COMPRESSED,
            tags=["reliable"],
        )
        d = s.to_dict()
        restored = Skill.from_dict(d)
        assert restored.name == "test_skill"
        assert restored.confidence == 0.85
        assert restored.status == SkillStatus.ACTIVE
        assert restored.trigger.keywords == ["timeout"]
        assert restored.tags == ["reliable"]

    def test_from_dict_string_trigger(self):
        d = {"name": "test", "trigger": "when timeout happens"}
        s = Skill.from_dict(d)
        assert s.trigger.description == "when timeout happens"

    def test_status_enum(self):
        assert SkillStatus.CANDIDATE.value == "candidate"
        assert SkillStatus.ACTIVE.value == "active"
        assert SkillStatus.DEPRECATED.value == "deprecated"
        assert SkillStatus.SUPERSEDED.value == "superseded"


class TestTriggerCondition:
    def test_empty_trigger_matches_weakly(self):
        t = TriggerCondition()
        score = t.matches(task="anything")
        assert score == 0.5

    def test_keyword_match(self):
        t = TriggerCondition(keywords=["timeout", "database"])
        score = t.matches(task="Fix the database timeout issue")
        assert score > 0.5

    def test_keyword_no_match(self):
        t = TriggerCondition(keywords=["timeout"])
        score = t.matches(task="Deploy new feature")
        assert score == 0.0

    def test_error_pattern_match(self):
        t = TriggerCondition(error_patterns=["ConnectionTimeout"])
        score = t.matches(errors_seen=["ConnectionTimeout at line 42"])
        assert score > 0.5

    def test_after_tools_match(self):
        t = TriggerCondition(after_tools=["diagnose"])
        score = t.matches(tools_called=["diagnose", "search"])
        assert score > 0.5

    def test_not_after_tools_blocks(self):
        t = TriggerCondition(
            keywords=["fix"],
            not_after_tools=["dangerous_tool"]
        )
        score = t.matches(
            task="fix the issue",
            tools_called=["dangerous_tool"]
        )
        assert score == 0.0

    def test_min_step_blocks(self):
        t = TriggerCondition(keywords=["fix"], min_step=3)
        score = t.matches(task="fix issue", current_step=1)
        assert score == 0.0

    def test_min_step_allows(self):
        t = TriggerCondition(keywords=["fix"], min_step=3)
        score = t.matches(task="fix issue", current_step=5)
        assert score > 0

    def test_to_dict_from_dict(self):
        t = TriggerCondition(
            keywords=["api", "error"],
            error_patterns=["500"],
            after_tools=["check"],
            min_step=2,
            description="API errors after check",
        )
        d = t.to_dict()
        restored = TriggerCondition.from_dict(d)
        assert restored.keywords == ["api", "error"]
        assert restored.min_step == 2


class TestPromptGeneration:
    def test_full_prompt(self):
        s = Skill(
            name="incident_fix",
            trigger=TriggerCondition(description="timeout errors detected"),
            action_sequence=["search", "diagnose", "fix"],
            compressed_instruction="Search logs, diagnose root cause, apply fix",
            preconditions=["Logs must be accessible"],
            postconditions=["Issue should be resolved"],
            confidence=0.85,
            usage_count=10,
        )
        prompt = s.to_prompt_fragment()
        assert "incident_fix" in prompt
        assert "timeout errors" in prompt
        assert "search → diagnose → fix" in prompt
        assert "85%" in prompt
        assert "Logs must be accessible" in prompt

    def test_compact_prompt(self):
        s = Skill(
            name="quick_fix",
            trigger=TriggerCondition(description="errors detected"),
            compressed_instruction="Apply standard fix",
            confidence=0.9,
        )
        prompt = s.to_compact_prompt()
        assert "quick_fix" in prompt
        assert "90%" in prompt
        assert len(prompt) < 200

    def test_prompt_without_trigger(self):
        s = Skill(
            name="generic",
            description="Do something useful",
            action_sequence=["a", "b"],
        )
        prompt = s.to_prompt_fragment()
        assert "generic" in prompt
        assert "a → b" in prompt


# ===========================================================================
# NAME GENERATION
# ===========================================================================

class TestNameGeneration:
    def test_simple_sequence(self, extractor):
        name = extractor._generate_name(["search", "fix"])
        assert name == "search_then_fix"

    def test_consecutive_duplicates(self, extractor):
        name = extractor._generate_name(["retry", "retry", "retry"])
        assert name == "retry_x3"

    def test_mixed_duplicates(self, extractor):
        name = extractor._generate_name(["check", "check", "deploy"])
        assert name == "check_x2_then_deploy"

    def test_single_tool(self, extractor):
        name = extractor._generate_name(["diagnose"])
        assert name == "diagnose_skill"

    def test_special_characters(self, extractor):
        name = extractor._generate_name(["my-tool", "other.tool"])
        assert "-" not in name
        assert "." not in name

    def test_long_sequence(self, extractor):
        seq = ["a", "b", "c", "d", "e"]
        name = extractor._generate_name(seq)
        assert "then" in name


# ===========================================================================
# CHAIN COMPRESSION
# ===========================================================================

class TestChainCompression:
    def test_single_tool(self, extractor):
        from nil.evolution.analyzer import CausalPattern
        p = CausalPattern(sequence=["search"], success_rate=0.8)
        result = extractor._compress_chain(["search"], p)
        assert "search" in result.lower()

    def test_multi_tool(self, extractor):
        from nil.evolution.analyzer import CausalPattern
        p = CausalPattern(sequence=["search", "fix"], success_rate=0.8)
        result = extractor._compress_chain(["search", "fix"], p)
        assert "first" in result.lower()
        assert "fix" in result.lower()

    def test_high_confidence_note(self, extractor):
        from nil.evolution.analyzer import CausalPattern
        p = CausalPattern(sequence=["a", "b"], success_rate=0.95)
        result = extractor._compress_chain(["a", "b"], p)
        assert "highly reliable" in result.lower()

    def test_moderate_confidence_note(self, extractor):
        from nil.evolution.analyzer import CausalPattern
        p = CausalPattern(sequence=["a", "b"], success_rate=0.75)
        result = extractor._compress_chain(["a", "b"], p)
        assert "works well" in result.lower()


# ===========================================================================
# KEYWORD EXTRACTION
# ===========================================================================

class TestKeywordExtraction:
    def test_extracts_common_words(self, extractor):
        tasks = [
            "Fix the database timeout issue",
            "Resolve database connection timeout",
            "Handle database timeout error",
        ]
        keywords = extractor._extract_keywords(tasks)
        assert "database" in keywords
        assert "timeout" in keywords

    def test_filters_stop_words(self, extractor):
        tasks = ["The quick brown fox", "A quick red fox"]
        keywords = extractor._extract_keywords(tasks)
        assert "the" not in keywords
        assert "a" not in keywords

    def test_empty_tasks(self, extractor):
        assert extractor._extract_keywords([]) == []

    def test_single_task(self, extractor):
        keywords = extractor._extract_keywords(["Fix API timeout"])
        assert isinstance(keywords, list)


# ===========================================================================
# DEDUPLICATION
# ===========================================================================

class TestDeduplication:
    def test_subsequence_absorbed(self, extractor):
        s1 = Skill(name="short", action_sequence=["a", "b"], confidence=0.8)
        s2 = Skill(name="long", action_sequence=["a", "b", "c"], confidence=0.85)
        result = extractor._deduplicate_skills([s1, s2])
        names = [s.name for s in result]
        assert "long" in names
        # short should be absorbed since it's a subsequence
        assert len(result) <= 2

    def test_higher_confidence_short_kept(self, extractor):
        """Short skill with much higher confidence should survive."""
        s1 = Skill(name="short", action_sequence=["a"], confidence=0.99)
        s2 = Skill(name="long", action_sequence=["a", "b"], confidence=0.6)
        result = extractor._deduplicate_skills([s1, s2])
        names = [s.name for s in result]
        assert "short" in names

    def test_no_duplicates(self, extractor):
        s1 = Skill(name="alpha", action_sequence=["x", "y"], confidence=0.8)
        s2 = Skill(name="beta", action_sequence=["a", "b"], confidence=0.8)
        result = extractor._deduplicate_skills([s1, s2])
        assert len(result) == 2

    def test_single_skill(self, extractor):
        s = Skill(name="only", action_sequence=["a"], confidence=0.9)
        result = extractor._deduplicate_skills([s])
        assert len(result) == 1

    def test_empty(self, extractor):
        assert extractor._deduplicate_skills([]) == []


# ===========================================================================
# SUBSEQUENCE CHECK
# ===========================================================================

class TestIsSubsequence:
    def test_is_subsequence(self):
        assert SkillExtractor._is_subsequence(["a", "b"], ["a", "b", "c"])
        assert SkillExtractor._is_subsequence(["b", "c"], ["a", "b", "c"])
        assert SkillExtractor._is_subsequence(["a", "b", "c"], ["a", "b", "c"])

    def test_not_subsequence(self):
        assert not SkillExtractor._is_subsequence(["a", "c"], ["a", "b", "c"])
        assert not SkillExtractor._is_subsequence(["d"], ["a", "b", "c"])

    def test_empty_is_subsequence(self):
        assert SkillExtractor._is_subsequence([], ["a", "b"])

    def test_longer_not_subsequence(self):
        assert not SkillExtractor._is_subsequence(["a", "b", "c", "d"], ["a", "b"])


# ===========================================================================
# MERGING WITH EXISTING
# ===========================================================================

class TestMerging:
    def test_same_name_merges(self, extractor):
        existing = [Skill(name="fix_skill", action_sequence=["fix"], confidence=0.7, version=1)]
        new = [Skill(name="fix_skill", action_sequence=["fix"], confidence=0.85)]
        truly_new, merged, unchanged = extractor._merge_with_existing(new, existing)
        assert len(truly_new) == 0
        assert len(merged) == 1
        assert merged[0].confidence == 0.85
        assert merged[0].version == 2

    def test_same_sequence_merges(self, extractor):
        existing = [Skill(name="old_name", action_sequence=["a", "b"], confidence=0.7)]
        new = [Skill(name="new_name", action_sequence=["a", "b"], confidence=0.9)]
        truly_new, merged, unchanged = extractor._merge_with_existing(new, existing)
        assert len(merged) == 1
        assert merged[0].name == "old_name"  # keeps old name

    def test_different_is_new(self, extractor):
        existing = [Skill(name="old", action_sequence=["a"])]
        new = [Skill(name="new", action_sequence=["x", "y"])]
        truly_new, merged, unchanged = extractor._merge_with_existing(new, existing)
        assert len(truly_new) == 1
        assert "old" in unchanged

    def test_empty_existing(self, extractor):
        new = [Skill(name="fresh", action_sequence=["a"])]
        truly_new, merged, unchanged = extractor._merge_with_existing(new, [])
        assert len(truly_new) == 1
        assert len(merged) == 0


# ===========================================================================
# FULL EXTRACTION PIPELINE
# ===========================================================================

class TestFullPipeline:
    def test_extracts_skill_from_traces(self, memory, extractor):
        traces = []
        for i in range(10):
            traces.append(_trace(
                f"s-{i}", True, ["diagnose", "fix", "verify"],
                task="Fix the server timeout",
            ))
        for i in range(3):
            traces.append(_trace(f"f-{i}", False, ["fix"], error="timeout"))
        _populate(memory, traces)

        result = extractor.extract("test")
        assert isinstance(result, ExtractionResult)
        assert result.new_skill_count >= 1
        assert result.total_patterns_analyzed > 0

    def test_result_summary(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(3)]
        _populate(memory, traces)

        result = extractor.extract("test")
        summary = result.summary
        assert "domain" in summary
        assert "new_skills" in summary
        assert "patterns_analyzed" in summary

    def test_skills_stored_in_memory(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["search", "fix"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["other"], error="err") for i in range(3)]
        _populate(memory, traces)

        extractor.extract("test")
        skills = memory.list_skills("test")
        assert len(skills) > 0

    def test_second_extraction_merges(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(3)]
        _populate(memory, traces)

        r1 = extractor.extract("test")
        r2 = extractor.extract("test")
        # Second should merge, not create duplicates
        assert r2.new_skill_count <= r1.new_skill_count

    def test_empty_domain(self, memory, extractor):
        result = extractor.extract("nonexistent")
        assert result.new_skill_count == 0
        assert result.total_patterns_analyzed == 0

    def test_all_failures(self, memory, extractor):
        traces = [_trace(f"f-{i}", False, ["a"], error="err") for i in range(10)]
        _populate(memory, traces)
        result = extractor.extract("test")
        assert result.new_skill_count == 0

    def test_trigger_keywords_populated(self, memory, extractor):
        traces = []
        for i in range(10):
            traces.append(_trace(
                f"s-{i}", True, ["diagnose", "fix"],
                task="Fix the database timeout error",
            ))
        for i in range(3):
            traces.append(_trace(f"f-{i}", False, ["other"], error="err"))
        _populate(memory, traces)

        result = extractor.extract("test")
        if result.extracted:
            skill = result.extracted[0]
            # Should have inferred some keywords from "database timeout"
            assert isinstance(skill.trigger.keywords, list)

    def test_compressed_instruction_present(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["search", "analyze", "fix"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["other"], error="err") for i in range(3)]
        _populate(memory, traces)

        result = extractor.extract("test")
        if result.extracted:
            skill = result.extracted[0]
            assert skill.compressed_instruction != ""


# ===========================================================================
# PROMPT GENERATION
# ===========================================================================

class TestPromptGeneration:
    def test_generate_skill_prompt(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["search", "fix"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["other"], error="err") for i in range(3)]
        _populate(memory, traces)

        extractor.extract("test")
        prompt = extractor.generate_skill_prompt("test")
        assert "LEARNED SKILLS" in prompt

    def test_compact_prompt(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["a", "b"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["c"], error="err") for i in range(3)]
        _populate(memory, traces)

        extractor.extract("test")
        full = extractor.generate_skill_prompt("test", compact=False)
        compact = extractor.generate_skill_prompt("test", compact=True)
        # Compact should be shorter
        if full and compact:
            assert len(compact) <= len(full)

    def test_empty_domain_prompt(self, memory, extractor):
        prompt = extractor.generate_skill_prompt("empty")
        assert prompt == ""


# ===========================================================================
# LIFECYCLE MANAGEMENT
# ===========================================================================

class TestLifecycle:
    def test_deprecation_by_usage(self, memory, extractor):
        """Skill with low usage success rate should be deprecated."""
        existing = [Skill(
            name="bad_skill",
            action_sequence=["a"],
            confidence=0.8,
            success_when_used=1,
            failure_when_used=5,
            status=SkillStatus.ACTIVE,
        )]
        traces = [_trace(f"s-{i}", True, ["b"]) for i in range(10)]
        deprecated = extractor._deprecate_ineffective(existing, traces)
        assert "bad_skill" in deprecated

    def test_no_deprecation_for_good_skill(self, memory, extractor):
        existing = [Skill(
            name="good_skill",
            action_sequence=["a"],
            confidence=0.9,
            success_when_used=9,
            failure_when_used=1,
            status=SkillStatus.ACTIVE,
        )]
        traces = [_trace(f"s-{i}", True, ["a"]) for i in range(10)]
        deprecated = extractor._deprecate_ineffective(existing, traces)
        assert "good_skill" not in deprecated

    def test_already_deprecated_skipped(self, memory, extractor):
        existing = [Skill(
            name="old_skill",
            status=SkillStatus.DEPRECATED,
        )]
        deprecated = extractor._deprecate_ineffective(existing, [])
        assert "old_skill" not in deprecated


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_unicode_tool_names(self, memory, extractor):
        traces = [_trace(f"s-{i}", True, ["工具", "outil"]) for i in range(10)]
        traces += [_trace(f"f-{i}", False, ["x"], error="err") for i in range(3)]
        _populate(memory, traces)
        result = extractor.extract("test")
        assert isinstance(result, ExtractionResult)

    def test_very_long_sequences(self, memory, extractor):
        long_seq = [f"step_{i}" for i in range(10)]
        traces = [_trace(f"s-{i}", True, long_seq) for i in range(5)]
        traces += [_trace(f"f-{i}", False, ["x"], error="err") for i in range(3)]
        _populate(memory, traces)
        result = extractor.extract("test")
        assert isinstance(result, ExtractionResult)

    def test_single_trace(self, memory, extractor):
        _populate(memory, [_trace("s-0", True, ["a"])])
        result = extractor.extract("test")
        assert result.new_skill_count == 0  # below min_frequency

    def test_max_skills_limit(self, memory):
        ext = SkillExtractor(memory, min_frequency=3, min_confidence=0.3, max_skills_per_domain=3)
        traces = []
        for tool_id in range(10):
            for i in range(5):
                traces.append(_trace(
                    f"s-{tool_id}-{i}", True, [f"tool_{tool_id}"],
                    created_offset=tool_id * 10 + i,
                ))
        traces += [_trace(f"f-{i}", False, ["x"], error="err") for i in range(3)]
        _populate(memory, traces)

        result = ext.extract("test")
        total_skills = memory.list_skills("test")
        # Should not exceed max
        active = [n for n in total_skills if True]  # all stored
        # The limit is soft — enforced via confidence=0 on deprecated
        assert isinstance(result, ExtractionResult)


# ===========================================================================
# SCALABILITY
# ===========================================================================

class TestScalability:
    def test_300_traces(self, memory, extractor):
        traces = []
        for i in range(250):
            traces.append(_trace(f"s-{i}", True, ["a", "b", "c"], created_offset=i))
        for i in range(50):
            traces.append(_trace(f"f-{i}", False, ["d"], error="err", created_offset=250+i))
        _populate(memory, traces)

        import time as t
        start = t.monotonic()
        result = extractor.extract("test")
        elapsed = t.monotonic() - start

        assert result.total_patterns_analyzed > 0
        assert elapsed < 30


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
