from nil.agents.messages import MessageBus
from nil.agents.orchestrator import OrchestratorAgent
from nil.agents.pool import AgentPool
from nil.agents.roles import AgentRole, AgentSpec, MultiAgentOutcome
from nil.agents.blackboard import Blackboard


class MultiAgentNil:
    def __init__(self, domain: str = ""):
        self.domain = domain
        self.agent_pool = AgentPool()
        self.blackboard = Blackboard()
        self.message_bus = MessageBus(agent_name="multi-agent-nil")
        self.orchestrator = OrchestratorAgent(
            domain=domain,
            agent_pool=self.agent_pool,
            blackboard=self.blackboard,
            name="orchestrator",
        )

    def register_agent_spec(self, spec: AgentSpec) -> None:
        self.agent_pool.register_spec(spec)

    def register_default_specs(self) -> None:
        self.register_agent_spec(
            AgentSpec(role=AgentRole.ORCHESTRATOR, name="orchestrator")
        )
        self.register_agent_spec(
            AgentSpec(role=AgentRole.INVESTIGATOR, name="investigator")
        )
        self.register_agent_spec(AgentSpec(role=AgentRole.ANALYST, name="analyst"))
        self.register_agent_spec(AgentSpec(role=AgentRole.EXECUTOR, name="executor"))
        self.register_agent_spec(AgentSpec(role=AgentRole.VERIFIER, name="verifier"))

    def run(self, task: str) -> MultiAgentOutcome:
        return self.orchestrator.run(task, {"domain": self.domain})
