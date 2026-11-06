"""Terminal coding agent that drives a model on the Boilerplate engine or any OpenAI-compatible endpoint.

See docs/agent_prompt_coding_cli.md for the design and security model.
"""

from .backends import AssistantTurn, LocalEngineBackend, MockBackend, OpenAIBackend
from .config import AgentConfig, load_config
from .loop import Agent, TurnResult
from .permissions import PermissionEngine
from .session import Session

__all__ = ["Agent", "AgentConfig", "AssistantTurn", "LocalEngineBackend", "MockBackend", "OpenAIBackend",
           "PermissionEngine", "Session", "TurnResult", "load_config"]
