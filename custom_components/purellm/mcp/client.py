"""MCP Protocol Client for PureLLM.

Implements the Model Context Protocol (MCP) client for connecting to
MCP-compatible servers that provide resources, tools, and prompts.

Supports both stdio-based and HTTP-based MCP server transports.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# MCP Protocol version
MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPError(Exception):
    """Base exception for MCP operations."""


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""


class MCPToolError(MCPError):
    """Error executing MCP tool."""


class MCPClient:
    """Client for connecting to MCP-compatible servers.

    Supports the Model Context Protocol for:
    - Resources: Read-only data sources (context, documents, memories)
    - Tools: Executable functions provided by the server
    - Prompts: Templated prompts with dynamic context

    This implementation uses HTTP transport for simplicity and
    compatibility with Home Assistant's async architecture.
    """

    def __init__(
        self,
        server_url: str,
        session: aiohttp.ClientSession,
        timeout: float = 30.0,
    ) -> None:
        """Initialize MCP client.

        Args:
            server_url: Base URL of the MCP server (e.g., http://localhost:3000)
            session: aiohttp ClientSession for making requests
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.session = session
        self.timeout = timeout

        # Server capabilities (populated after initialize)
        self._capabilities: dict[str, Any] = {}
        self._server_info: dict[str, Any] = {}
        self._connected = False

        # Cache for resources and tools
        self._resources_cache: list[dict] | None = None
        self._tools_cache: list[dict] | None = None

    @property
    def is_connected(self) -> bool:
        """Return True if connected to server."""
        return self._connected

    @property
    def server_name(self) -> str:
        """Return server name if connected."""
        return self._server_info.get("name", "Unknown")

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return server capabilities."""
        return self._capabilities

    async def connect(self) -> bool:
        """Initialize connection and exchange capabilities.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            _LOGGER.info("Connecting to MCP server at %s", self.server_url)

            # Send initialize request
            response = await self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {},
                    },
                    "clientInfo": {
                        "name": "purellm",
                        "version": "5.3.0",
                    },
                },
            )

            if not response:
                _LOGGER.error("Empty response from MCP server initialize")
                return False

            # Store server capabilities
            self._capabilities = response.get("capabilities", {})
            self._server_info = response.get("serverInfo", {})
            self._connected = True

            _LOGGER.info(
                "Connected to MCP server: %s (protocol: %s)",
                self._server_info.get("name", "Unknown"),
                response.get("protocolVersion", "unknown"),
            )

            # Send initialized notification
            await self._notify("notifications/initialized", {})

            return True

        except Exception as err:
            _LOGGER.error("Failed to connect to MCP server: %s", err)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from MCP server."""
        if self._connected:
            _LOGGER.info("Disconnecting from MCP server")
            self._connected = False
            self._resources_cache = None
            self._tools_cache = None

    async def list_resources(self, force_refresh: bool = False) -> list[dict]:
        """List available resources from the server.

        Args:
            force_refresh: If True, bypass cache and fetch from server

        Returns:
            List of resource definitions with uri, name, description, mimeType
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        if not force_refresh and self._resources_cache is not None:
            return self._resources_cache

        response = await self._request("resources/list", {})
        self._resources_cache = response.get("resources", [])

        _LOGGER.debug("Listed %d MCP resources", len(self._resources_cache))
        return self._resources_cache

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read content from a resource.

        Args:
            uri: Resource URI (e.g., memory://conversations/recent)

        Returns:
            Resource content with uri and contents array
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        response = await self._request("resources/read", {"uri": uri})
        return response

    async def list_tools(self, force_refresh: bool = False) -> list[dict]:
        """List available tools from the server.

        Args:
            force_refresh: If True, bypass cache and fetch from server

        Returns:
            List of tool definitions with name, description, inputSchema
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        if not force_refresh and self._tools_cache is not None:
            return self._tools_cache

        response = await self._request("tools/list", {})
        self._tools_cache = response.get("tools", [])

        _LOGGER.debug("Listed %d MCP tools", len(self._tools_cache))
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> list[dict]:
        """Execute a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            List of content blocks (text, image, resource)
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        _LOGGER.debug("Calling MCP tool: %s(%s)", name, arguments)

        try:
            response = await self._request(
                "tools/call",
                {"name": name, "arguments": arguments},
            )

            if response.get("isError"):
                error_content = response.get("content", [])
                error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
                raise MCPToolError(f"Tool {name} failed: {error_text}")

            return response.get("content", [])

        except MCPToolError:
            raise
        except Exception as err:
            raise MCPToolError(f"Failed to call tool {name}: {err}") from err

    async def list_prompts(self) -> list[dict]:
        """List available prompts from the server.

        Returns:
            List of prompt definitions with name, description, arguments
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        response = await self._request("prompts/list", {})
        return response.get("prompts", [])

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Get a prompt with optional arguments.

        Args:
            name: Prompt name
            arguments: Optional prompt arguments

        Returns:
            Prompt content with description and messages
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server")

        response = await self._request(
            "prompts/get",
            {"name": name, "arguments": arguments or {}},
        )
        return response

    async def ping(self) -> bool:
        """Check if server is responsive.

        Returns:
            True if server responds to ping
        """
        try:
            await self._request("ping", {})
            return True
        except Exception:
            return False

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send JSON-RPC request to MCP server.

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            Response result dict
        """
        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        try:
            async with self.session.post(
                f"{self.server_url}/message",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise MCPError(f"MCP request failed ({resp.status}): {error_text}")

                data = await resp.json()

                if "error" in data:
                    error = data["error"]
                    raise MCPError(f"MCP error {error.get('code')}: {error.get('message')}")

                return data.get("result", {})

        except asyncio.TimeoutError as err:
            raise MCPConnectionError(f"MCP request timed out: {method}") from err

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send JSON-RPC notification (no response expected).

        Args:
            method: Notification method name
            params: Notification parameters
        """
        import aiohttp

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            async with self.session.post(
                f"{self.server_url}/message",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5),
                headers={"Content-Type": "application/json"},
            ) as resp:
                # Notifications don't require response checking
                pass
        except Exception as err:
            _LOGGER.debug("MCP notification failed (non-critical): %s", err)

    def get_tools_as_openai_format(self, tools: list[dict]) -> list[dict]:
        """Convert MCP tools to OpenAI function calling format.

        Args:
            tools: List of MCP tool definitions

        Returns:
            List of OpenAI-compatible tool definitions
        """
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools
