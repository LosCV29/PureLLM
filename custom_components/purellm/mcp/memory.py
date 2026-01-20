"""MCP Memory Manager for PureLLM.

Provides high-level memory operations through self-hosted mcp-mem0 server.
Uses Docker-hosted mem0 with PostgreSQL + pgvector for semantic memory.

Memory Flow:
1. Before LLM call: Retrieve relevant memories based on user query
2. Inject memories into system prompt context
3. After response: Extract and store important facts/preferences

Server: https://github.com/coleam00/mcp-mem0 (self-hosted)
"""
from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import MCPClient

_LOGGER = logging.getLogger(__name__)


class MCPMemoryManager:
    """Memory manager using self-hosted mcp-mem0 server.

    Connects to a Docker-hosted mcp-mem0 server that provides:
    - save_memory: Store information with semantic indexing
    - search_memories: Find relevant memories using semantic search
    - get_all_memories: Retrieve all stored memories

    Provides:
    - Context retrieval for query-relevant memories
    - Memory storage for facts, preferences, and conversation summaries
    - Automatic memory formatting for LLM context injection
    """

    # Memory types for categorization
    MEMORY_TYPE_FACT = "fact"
    MEMORY_TYPE_PREFERENCE = "preference"
    MEMORY_TYPE_CONVERSATION = "conversation"
    MEMORY_TYPE_INSTRUCTION = "instruction"

    def __init__(
        self,
        client: MCPClient,
        user_id: str = "purellm_user",
        max_context_memories: int = 5,
    ) -> None:
        """Initialize memory manager.

        Args:
            client: Connected MCPClient instance
            user_id: User identifier for memory scoping
            max_context_memories: Max memories to inject into context
        """
        self.client = client
        self.user_id = user_id
        self.max_context_memories = max_context_memories
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize and verify server has required tools.

        Returns:
            True if memory server is ready
        """
        try:
            tools = await self.client.list_tools()
            tool_names = {t.get("name", "") for t in tools}

            # Verify self-hosted mcp-mem0 tools are available
            required = {"save_memory", "search_memories", "get_all_memories"}
            if not required.issubset(tool_names):
                missing = required - tool_names
                _LOGGER.error("MCP memory server missing tools: %s", missing)
                return False

            self._initialized = True
            _LOGGER.info("MCP memory manager initialized with tools: %s", tool_names)
            return True

        except Exception as err:
            _LOGGER.error("Failed to initialize memory manager: %s", err)
            return False

    @property
    def is_ready(self) -> bool:
        """Return True if memory manager is initialized and ready."""
        return self._initialized and self.client.is_connected

    async def store_memory(
        self,
        content: str,
        memory_type: str = MEMORY_TYPE_FACT,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a memory in the MCP server.

        Args:
            content: Memory content to store
            memory_type: Type of memory (fact, preference, conversation, instruction)
            metadata: Optional metadata to attach

        Returns:
            True if stored successfully
        """
        if not self.is_ready:
            _LOGGER.warning("Memory manager not ready, skipping store")
            return False

        try:
            # Build metadata
            meta = metadata or {}
            meta["type"] = memory_type
            meta["source"] = "purellm"
            meta["user_id"] = self.user_id

            # Self-hosted mcp-mem0: save_memory tool
            await self.client.call_tool(
                "save_memory",
                {
                    "content": content,
                    "metadata": meta,
                },
            )

            _LOGGER.debug("Stored memory: %s...", content[:50])
            return True

        except Exception as err:
            _LOGGER.error("Failed to store memory: %s", err)
            return False

    async def search_memories(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories semantically.

        Args:
            query: Search query
            limit: Max results (defaults to max_context_memories)

        Returns:
            List of memory dicts with content and metadata
        """
        if not self.is_ready:
            return []

        limit = limit or self.max_context_memories

        try:
            # Self-hosted mcp-mem0: search_memories tool
            result = await self.client.call_tool(
                "search_memories",
                {
                    "query": query,
                    "limit": limit,
                },
            )

            # Parse results from MCP content blocks
            memories = self._parse_memory_results(result)
            _LOGGER.debug("Found %d memories for query: %s", len(memories), query[:30])
            return memories

        except Exception as err:
            _LOGGER.error("Failed to search memories: %s", err)
            return []

    async def get_all_memories(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get all memories (for debugging/admin).

        Args:
            limit: Max memories to return

        Returns:
            List of all memories
        """
        if not self.is_ready:
            return []

        try:
            # Self-hosted mcp-mem0: get_all_memories tool
            result = await self.client.call_tool(
                "get_all_memories",
                {},
            )

            memories = self._parse_memory_results(result)
            # Apply limit client-side since get_all_memories may not support it
            return memories[:limit]

        except Exception as err:
            _LOGGER.error("Failed to get all memories: %s", err)
            return []

    async def get_context_for_query(self, query: str) -> str:
        """Get formatted memory context for a user query.

        This is the main method called before LLM requests to inject
        relevant memories into the conversation context.

        Args:
            query: User's input query

        Returns:
            Formatted memory context string (empty if no memories)
        """
        memories = await self.search_memories(query)

        if not memories:
            return ""

        # Format memories for context injection
        lines = ["## Remembered Context"]
        lines.append("The following information was remembered from previous interactions:")
        lines.append("")

        for i, mem in enumerate(memories, 1):
            content = mem.get("content") or mem.get("text") or mem.get("memory", "")
            if content:
                # Clean up content
                content = content.strip()
                if not content.endswith("."):
                    content += "."
                lines.append(f"- {content}")

        lines.append("")
        return "\n".join(lines)

    async def extract_and_store_memories(
        self,
        user_query: str,
        assistant_response: str,
        force: bool = False,
    ) -> int:
        """Extract important information and store as memories.

        Analyzes the conversation to identify facts worth remembering:
        - User preferences (likes, dislikes, settings)
        - Personal facts (names, locations, relationships)
        - Instructions (how they want things done)

        Args:
            user_query: What the user asked
            assistant_response: What was responded
            force: Store even if not deemed important

        Returns:
            Number of memories stored
        """
        # Simple heuristic extraction - in production, use LLM for extraction
        memories_to_store = []

        # Detect preference statements
        preference_patterns = [
            r"(?:i |my )(?:prefer|like|love|hate|don't like|always want|never want)",
            r"(?:please |always |never )(?:use|set|make|keep)",
            r"(?:call me|my name is|i am|i'm called)",
        ]

        query_lower = user_query.lower()
        for pattern in preference_patterns:
            if re.search(pattern, query_lower):
                memories_to_store.append({
                    "content": f"User said: {user_query}",
                    "type": self.MEMORY_TYPE_PREFERENCE,
                })
                break

        # Detect explicit memory requests
        if any(phrase in query_lower for phrase in ["remember that", "don't forget", "keep in mind"]):
            memories_to_store.append({
                "content": user_query,
                "type": self.MEMORY_TYPE_INSTRUCTION,
            })

        # Force store if requested (useful for explicit "remember this" commands)
        if force and not memories_to_store:
            memories_to_store.append({
                "content": f"Conversation: User asked '{user_query}', assistant responded about {assistant_response[:100]}...",
                "type": self.MEMORY_TYPE_CONVERSATION,
            })

        # Store extracted memories
        stored = 0
        for mem in memories_to_store:
            if await self.store_memory(mem["content"], mem["type"]):
                stored += 1

        if stored > 0:
            _LOGGER.info("Extracted and stored %d memories", stored)

        return stored

    def _parse_memory_results(self, result: list[dict]) -> list[dict[str, Any]]:
        """Parse MCP tool result into memory dicts.

        Args:
            result: MCP content blocks from tool call

        Returns:
            List of memory dicts
        """
        memories = []

        for block in result:
            if block.get("type") == "text":
                text = block.get("text", "")

                # Try to parse as JSON (some servers return JSON)
                try:
                    import json
                    data = json.loads(text)
                    if isinstance(data, list):
                        memories.extend(data)
                    elif isinstance(data, dict):
                        if "memories" in data:
                            memories.extend(data["memories"])
                        elif "results" in data:
                            memories.extend(data["results"])
                        else:
                            memories.append(data)
                except (json.JSONDecodeError, TypeError):
                    # Plain text memory
                    if text.strip():
                        memories.append({"content": text})

        return memories


class MemoryTool:
    """Tool handler for explicit memory operations via voice commands.

    Allows users to explicitly manage memories:
    - "Remember that I prefer warm lighting"
    - "What do you remember about me?"
    - "Forget everything about my schedule"
    """

    def __init__(self, memory_manager: MCPMemoryManager) -> None:
        """Initialize tool with memory manager."""
        self.memory = memory_manager

    @staticmethod
    def get_tool_definitions() -> list[dict]:
        """Return OpenAI-format tool definitions for memory operations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember_information",
                    "description": "Store information for future reference. Use when user says 'remember that...' or explicitly asks to save something.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The information to remember"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["preference", "fact", "instruction"],
                                "description": "Type of information: preference (likes/dislikes), fact (personal info), instruction (how to do things)"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_memories",
                    "description": "Search for remembered information. Use when user asks 'what do you remember about...' or 'do you know my...'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What to search for in memories"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def remember_information(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle remember_information tool call."""
        content = arguments.get("content", "")
        category = arguments.get("category", MCPMemoryManager.MEMORY_TYPE_FACT)

        if not content:
            return {"error": "No content provided to remember"}

        success = await self.memory.store_memory(content, category)

        if success:
            return {
                "success": True,
                "response_text": f"I'll remember that."
            }
        else:
            return {
                "success": False,
                "response_text": "I wasn't able to save that, but I'll keep it in mind for now."
            }

    async def recall_memories(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle recall_memories tool call."""
        query = arguments.get("query", "")

        if not query:
            # Get all memories
            memories = await self.memory.get_all_memories(limit=10)
        else:
            memories = await self.memory.search_memories(query, limit=5)

        if not memories:
            return {
                "memories": [],
                "response_text": "I don't have any memories about that yet."
            }

        # Format memories for response
        memory_texts = []
        for mem in memories:
            content = mem.get("content") or mem.get("text") or mem.get("memory", "")
            if content:
                memory_texts.append(content)

        return {
            "memories": memory_texts,
            "count": len(memory_texts),
            "response_text": f"I remember {len(memory_texts)} thing(s): " + "; ".join(memory_texts[:3])
        }
