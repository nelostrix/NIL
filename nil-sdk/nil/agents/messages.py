from __future__ import annotations

import enum
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class MessageType(enum.Enum):
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    DELEGATE = "delegate"
    VOTE = "vote"
    BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    source: str = ""
    target: str = ""
    message_type: MessageType = MessageType.INFORM
    content: Any = None
    thread_id: str = ""
    priority: int = 0
    deadline: float = 0.0
    correlation_id: str = ""

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = uuid.uuid4().hex[:12]
        if not self.correlation_id:
            self.correlation_id = uuid.uuid4().hex[:12]
        if not self.deadline:
            self.deadline = time.time() + 30


@dataclass
class AgentResponse:
    content: Any = None
    success: bool = True
    error: str = ""

    @classmethod
    def error(cls, msg: str) -> AgentResponse:
        return cls(success=False, error=msg)


class MessageBus:
    def __init__(self, agent_name: str = ""):
        self.agent_name = agent_name
        self.queues: dict[str, list] = defaultdict(list)
        self.topics: dict[str, list[Callable]] = defaultdict(list)
        self._pending_requests: dict[str, Callable] = {}
        self._history: list[AgentMessage] = []

    def send(self, message: AgentMessage) -> None:
        self._history.append(message)
        if message.target == "broadcast":
            self._broadcast(message)
        elif message.target in self.queues:
            self.queues[message.target].append(message)
        if message.correlation_id in self._pending_requests:
            cb = self._pending_requests.pop(message.correlation_id)
            cb(message)

    def request(self, target: str, content: Any, timeout: float = 30) -> Any:
        correlation_id = uuid.uuid4().hex[:12]
        message = AgentMessage(
            source=self.agent_name,
            target=target,
            message_type=MessageType.REQUEST,
            content=content,
            correlation_id=correlation_id,
        )
        result = {"content": None, "received": False}

        def callback(msg: AgentMessage) -> None:
            result["content"] = msg.content
            result["received"] = True

        self._pending_requests[correlation_id] = callback
        self.send(message)
        return result

    def respond(self, request: AgentMessage, content: Any) -> None:
        response = AgentMessage(
            source=self.agent_name,
            target=request.source,
            message_type=MessageType.RESPONSE,
            content=content,
            correlation_id=request.correlation_id,
            thread_id=request.thread_id,
        )
        self.send(response)

    def subscribe(self, topic: str, callback: Callable) -> None:
        self.topics[topic].append(callback)

    def publish(self, topic: str, data: Any) -> None:
        for cb in self.topics.get(topic, []):
            cb(data)

    def receive(self, agent_name: str) -> list[AgentMessage]:
        messages = list(self.queues.get(agent_name, []))
        self.queues[agent_name] = []
        return messages

    def _broadcast(self, message: AgentMessage) -> None:
        for agent_name in self.queues:
            if agent_name != message.source:
                self.queues[agent_name].append(message)
