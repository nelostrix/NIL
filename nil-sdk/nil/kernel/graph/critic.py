from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from nil.kernel.graph.graph_types import ExecutionNode, NodeResult

logger = logging.getLogger("nil.kernel.graph.critic")


class CritiqueVerdict(str, Enum):
    PASS = "pass"
    NEEDS_REVISION = "needs_revision"
    FAIL = "fail"


@dataclass
class CritiqueIssue:
    description: str
    severity: str = "medium"
    suggested_action: str = ""
    node_id: str = ""

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "severity": self.severity,
            "suggested_action": self.suggested_action,
            "node_id": self.node_id,
        }


@dataclass
class CritiqueResult:
    verdict: CritiqueVerdict
    issues: list[CritiqueIssue] = field(default_factory=list)
    summary: str = ""
    confidence: float = 1.0

    @property
    def passed(self) -> bool:
        return self.verdict == CritiqueVerdict.PASS

    @property
    def needs_revision(self) -> bool:
        return self.verdict == CritiqueVerdict.NEEDS_REVISION

    @property
    def failed(self) -> bool:
        return self.verdict == CritiqueVerdict.FAIL

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "confidence": self.confidence,
        }


class Critic:
    def __init__(self):
        self._validators: list[CritiqueRule] = []

    def add_validator(self, rule: CritiqueRule) -> None:
        self._validators.append(rule)

    def evaluate(
        self,
        node: ExecutionNode,
        node_output: Any,
        context: dict[str, Any] = None,
    ) -> CritiqueResult:
        all_issues: list[CritiqueIssue] = []
        for validator in self._validators:
            try:
                result = validator.check(node, node_output, context or {})
                if result:
                    all_issues.extend(result)
            except Exception as e:
                logger.warning("Validator %s error: %s", type(validator).__name__, e)

        if not all_issues:
            return CritiqueResult(
                verdict=CritiqueVerdict.PASS,
                summary="All validation checks passed",
            )

        has_critical = any(i.severity == "critical" for i in all_issues)
        has_high = any(i.severity in ("critical", "high") for i in all_issues)

        if has_critical:
            verdict = CritiqueVerdict.FAIL
        elif has_high:
            verdict = CritiqueVerdict.NEEDS_REVISION
        else:
            verdict = CritiqueVerdict.PASS

        return CritiqueResult(
            verdict=verdict,
            issues=all_issues,
            summary=f"Found {len(all_issues)} issue(s): {', '.join(i.description[:50] for i in all_issues[:3])}",
            confidence=1.0 - (len(all_issues) * 0.1),
        )

    def evaluate_output(
        self,
        output: Any,
        node_id: str = "",
    ) -> CritiqueResult:
        issues: list[CritiqueIssue] = []
        if output is None:
            issues.append(CritiqueIssue(
                description="Output is None",
                severity="high",
                node_id=node_id,
            ))
        if isinstance(output, dict) and not output:
            issues.append(CritiqueIssue(
                description="Output is an empty dict",
                severity="low",
                node_id=node_id,
            ))
        if isinstance(output, str) and len(output) < 5:
            issues.append(CritiqueIssue(
                description="Output is too short to be meaningful",
                severity="low",
                node_id=node_id,
            ))

        if not issues:
            return CritiqueResult(verdict=CritiqueVerdict.PASS, summary="Output looks valid")
        return CritiqueResult(
            verdict=CritiqueVerdict.NEEDS_REVISION if len(issues) <= 2 else CritiqueVerdict.FAIL,
            issues=issues,
            summary=f"Output validation found {len(issues)} issues",
            confidence=1.0 - (len(issues) * 0.1),
        )


class CritiqueRule:
    def check(
        self, node: ExecutionNode, output: Any, context: dict[str, Any]
    ) -> list[CritiqueIssue]:
        raise NotImplementedError


class SchemaValidator(CritiqueRule):
    def check(
        self, node: ExecutionNode, output: Any, context: dict[str, Any]
    ) -> list[CritiqueIssue]:
        issues = []
        schema = node.output_schema
        if not schema:
            return issues
        if not isinstance(output, dict):
            return issues
        for key, expected_type in schema.items():
            if key not in output:
                issues.append(CritiqueIssue(
                    description=f"Missing required output field: {key}",
                    severity="high",
                    suggested_action=f"Ensure output contains '{key}'",
                    node_id=node.node_id,
                ))
            elif expected_type == "str" and not isinstance(output.get(key), str):
                issues.append(CritiqueIssue(
                    description=f"Field '{key}' should be str, got {type(output.get(key)).__name__}",
                    severity="medium",
                    node_id=node.node_id,
                ))
        return issues


class ErrorFreeValidator(CritiqueRule):
    def check(
        self, node: ExecutionNode, output: Any, context: dict[str, Any]
    ) -> list[CritiqueIssue]:
        if isinstance(output, dict) and output.get("error"):
            return [CritiqueIssue(
                description=f"Node produced error: {output['error'][:100]}",
                severity="critical",
                node_id=node.node_id,
            )]
        if isinstance(output, str) and output.lower().startswith("error"):
            return [CritiqueIssue(
                description=f"Output starts with error: {output[:100]}",
                severity="high",
                node_id=node.node_id,
            )]
        return []
