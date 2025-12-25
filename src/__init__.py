"""Agentic AI Template - Multi-agent framework using LangGraph."""

from src.agents.base import BaseAgent
from src.agents.supervisor import SupervisorAgent
from src.config import settings
from src.graph.workflow import create_workflow

__all__ = ["BaseAgent", "SupervisorAgent", "create_workflow", "settings"]
__version__ = "1.0.0"
