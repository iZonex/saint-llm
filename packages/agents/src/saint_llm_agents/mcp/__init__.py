"""MCP (Model Context Protocol) adapter for saint-llm agents.

Two halves:

* :class:`MCPServer` — wraps a :class:`ToolRegistry` and serves it
  over the MCP JSON-RPC dialect (``initialize`` / ``tools/list`` /
  ``tools/call``). Lets external MCP clients (Claude Desktop, IDE
  plugins, other agents) call our tools via the standard protocol.
* :func:`mcp_client_tools` — connects to an external MCP server and
  returns its tools as local :class:`Tool` objects, ready to register
  in a saint-llm :class:`ToolRegistry`.

Transport is abstracted via :class:`JsonRpcChannel` so tests can run
fully in-process. A :class:`StdioJsonRpcChannel` is provided for the
production path (spawn an MCP server subprocess and talk over its
stdio).
"""

from saint_llm_agents.mcp.channel import (
    InMemoryJsonRpcChannel,
    JsonRpcChannel,
    StdioJsonRpcChannel,
    pair_channels,
)
from saint_llm_agents.mcp.client import mcp_client_tools, mcp_list_tools
from saint_llm_agents.mcp.server import MCPServer

__all__ = [
    "InMemoryJsonRpcChannel",
    "JsonRpcChannel",
    "MCPServer",
    "StdioJsonRpcChannel",
    "mcp_client_tools",
    "mcp_list_tools",
    "pair_channels",
]
