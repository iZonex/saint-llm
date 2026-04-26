"""Saint LLM agents — multi-agent runtime.

Modules:
    message  — Message + ToolCall dataclasses (frozen, OpenAI/Anthropic-shaped)
    tool     — ToolSpec + Tool protocol + FunctionTool + ToolRegistry
    policy   — Policy protocol + MockPolicy + CallablePolicy
    agent    — Agent class with step / run loop driving think→act→observe
    runtime  — Round-robin Runtime with per-agent inboxes + transcript
"""

from saint_llm_agents.a2a import (
    A2AClient,
    A2AClientError,
    A2AServer,
    A2ATask,
    InMemoryTaskStore,
    send_to_agent,
)
from saint_llm_agents.agent import Agent
from saint_llm_agents.code_mode import (
    ApiStub,
    CodeModePolicy,
    ExecutionResult,
    Executor,
    InProcessExecutor,
    generate_api_stub,
)
from saint_llm_agents.mcp import (
    MCPServer,
    mcp_client_tools,
    mcp_list_tools,
)
from saint_llm_agents.memory import (
    FileMemoryStore,
    InMemoryMemoryStore,
    MemoryEntry,
    MemoryStore,
    memory_tools,
    register_memory_tools,
)
from saint_llm_agents.lm_policy import TextLMPolicy
from saint_llm_agents.message import Message, ToolCall
from saint_llm_agents.policy import CallablePolicy, MockPolicy, Policy
from saint_llm_agents.runtime import Runtime
from saint_llm_agents.skills import (
    ScriptResult,
    ScriptRunner,
    Skill,
    SkillLoadError,
    SkillRegistry,
    load_skill,
    load_skills,
    register_skill_scripts,
    register_skills_as_tools,
)
from saint_llm_agents.tool import FunctionTool, Tool, ToolRegistry, ToolSpec
from saint_llm_agents.tool_call_parser import parse_tool_calls

__version__ = "0.0.1"

__all__ = [
    "A2AClient",
    "A2AClientError",
    "A2AServer",
    "A2ATask",
    "Agent",
    "ApiStub",
    "CallablePolicy",
    "CodeModePolicy",
    "ExecutionResult",
    "Executor",
    "FileMemoryStore",
    "FunctionTool",
    "InMemoryMemoryStore",
    "InMemoryTaskStore",
    "InProcessExecutor",
    "MCPServer",
    "MemoryEntry",
    "MemoryStore",
    "Message",
    "MockPolicy",
    "Policy",
    "Runtime",
    "ScriptResult",
    "ScriptRunner",
    "Skill",
    "SkillLoadError",
    "SkillRegistry",
    "TextLMPolicy",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "generate_api_stub",
    "load_skill",
    "load_skills",
    "mcp_client_tools",
    "mcp_list_tools",
    "memory_tools",
    "parse_tool_calls",
    "register_memory_tools",
    "register_skill_scripts",
    "register_skills_as_tools",
    "send_to_agent",
]
