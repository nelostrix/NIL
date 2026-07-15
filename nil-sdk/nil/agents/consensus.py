from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from nil.agents.roles import AgentRole, AgentSpec


@dataclass
class AgentProposal:
    id: str = ""
    label: str = ""
    value: Any = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]


@dataclass
class AgentOpinion:
    value: float = 0.0
    confidence: float = 0.5
    agent_name: str = ""


@dataclass
class ConsensusResult:
    winner: str = ""
    value: float = 0.0
    vote_share: float = 0.0
    confidence: float = 0.0
    n_agents: int = 0
    label: str = ""

    def is_conclusive(self, threshold: float = 0.5) -> bool:
        return self.vote_share >= threshold or self.confidence >= threshold


@dataclass
class DebateOutcome:
    winner: str = ""
    confidence: float = 0.0
    transcript: list[dict] = field(default_factory=list)
    label: str = ""


class ConsensusMechanism:
    def __init__(self):
        self._debate_log: list[dict] = []

    def majority_vote(
        self,
        proposals: list[AgentProposal],
        voters: list[AgentSpec],
        voter_evaluations: dict[str, str] = None,
    ) -> ConsensusResult:
        voter_evaluations = voter_evaluations or {}
        votes: dict[str, int] = {}
        for p in proposals:
            votes[p.id] = 0

        for voter in voters:
            chosen = voter_evaluations.get(voter.name) or voter_evaluations.get(
                voter.role.value
            )
            if chosen and chosen in votes:
                votes[chosen] += 1
            elif proposals:
                votes[proposals[0].id] += 1

        if not votes:
            return ConsensusResult()

        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        vote_share = votes[winner] / max(total, 1)

        return ConsensusResult(
            winner=winner,
            vote_share=vote_share,
            confidence=self._compute_confidence(votes, voters),
            n_agents=len(voters),
            label=next((p.label for p in proposals if p.id == winner), ""),
        )

    def debate(
        self,
        thesis: str,
        antithesis: str,
        rounds: int = 3,
        proponent_name: str = "proponent",
        opponent_name: str = "opponent",
        judge_name: str = "judge",
    ) -> DebateOutcome:
        self._debate_log = []
        final_score = 0.5

        for round_num in range(rounds):
            pro_argument = f"Argue for: {thesis} (round {round_num})"
            con_argument = f"Argue against: {thesis} (round {round_num})"
            score = 0.5 + (round_num * 0.1) / rounds

            self._debate_log.append(
                {
                    "round": round_num,
                    "pro_argument": pro_argument,
                    "con_argument": con_argument,
                    "score": score,
                }
            )
            final_score = score

        return DebateOutcome(
            winner=proponent_name if final_score > 0.5 else opponent_name,
            confidence=abs(final_score - 0.5) * 2,
            transcript=list(self._debate_log),
            label=thesis if final_score > 0.5 else antithesis,
        )

    def confidence_weighted_consensus(
        self,
        opinions: list[AgentOpinion],
    ) -> ConsensusResult:
        if not opinions:
            return ConsensusResult()

        total_weight = sum(o.confidence for o in opinions)
        if total_weight == 0:
            return ConsensusResult()

        weighted_sum = sum(o.value * o.confidence for o in opinions)
        average = weighted_sum / total_weight

        variance = (
            sum(o.confidence * (o.value - average) ** 2 for o in opinions)
            / total_weight
        )

        return ConsensusResult(
            value=average,
            confidence=1.0 / (1.0 + variance) if variance > 0 else 1.0,
            n_agents=len(opinions),
        )

    def _compute_confidence(
        self, votes: dict[str, int], voters: list[AgentSpec]
    ) -> float:
        if not votes or not voters:
            return 0.0
        total = sum(votes.values())
        if total == 0:
            return 0.0
        max_share = max(votes.values()) / total
        return max_share
