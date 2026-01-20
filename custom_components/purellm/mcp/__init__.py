"""MCP (Model Context Protocol) integration for PureLLM.

Provides persistent memory and context capabilities through MCP-compatible servers.
"""
from .client import MCPClient
from .memory import MCPMemoryManager

__all__ = ["MCPClient", "MCPMemoryManager"]
