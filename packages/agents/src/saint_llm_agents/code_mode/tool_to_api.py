"""Generate Python API stub from a ToolRegistry — Code Mode (ADR-0013).

Cloudflare/Anthropic Code Mode pattern: instead of describing each tool
as a JSON-Schema-tagged tool-call envelope, expose tools as a Python
namespace the model can call from generated code. The model writes a
short script; the runtime executes it in a sandbox; results come back.

Wins (per ADR-0013):
* 98%+ input-token reduction at production (no schema enumeration in
  prompt).
* 30-40% latency drop (single round-trip for multi-step tool use).
* Composes naturally with sandbox tiers + Anthropic Skills.

This module is the **prompt-side** of Code Mode: take a registry,
emit the Python API stub the model sees. Sandbox execution lives in
:mod:`saint_llm_agents.code_mode.sandbox_executor`.
"""

from __future__ import annotations

from dataclasses import dataclass

from saint_llm_agents.tool import ToolSpec


@dataclass(frozen=True)
class ApiStub:
    """One generated tool-API namespace stub.

    Attributes:
        source: Python source-text declaring functions matching the
            tool registry. Each function has a docstring derived from
            the tool description and the JSON Schema parameters.
        names:  the function names exposed (matches ``ToolSpec.name``).
    """

    source: str
    names: tuple[str, ...]


def _format_param_doc(parameters: dict) -> str:
    """Render a JSON Schema parameters dict as a docstring section."""
    if not parameters:
        return "    No parameters.\n"
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    if not props:
        return "    No parameters.\n"
    lines = ["    Args:"]
    for name, schema in props.items():
        type_hint = schema.get("type", "any")
        desc = schema.get("description", "")
        is_req = " (required)" if name in required else ""
        line = f"        {name} ({type_hint}){is_req}"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def generate_api_stub(specs: list[ToolSpec], *, namespace: str = "tools") -> ApiStub:
    """Render a Python API stub from a list of ``ToolSpec``.

    The result is a string of valid Python — when executed in the
    sandbox, each function makes a runtime call back into the
    registry to invoke the underlying tool. The runtime substitutes
    these stub bodies; the model only sees the namespace shape.

    Example output for a registry with ``add(a, b)``::

        class tools:
            @staticmethod
            def add(**kwargs):
                \"\"\"add two integers.

                Args:
                    a (integer) (required)
                    b (integer) (required)
                \"\"\"
                ...  # runtime-bound

    Args:
        specs:     tool specs to render.
        namespace: name of the wrapping class. Default ``"tools"`` so
            the model writes ``tools.add(a=2, b=3)``.

    Returns:
        :class:`ApiStub` with the rendered ``source`` and the tuple of
        exposed function ``names``.
    """
    if not specs:
        return ApiStub(source=f"class {namespace}:\n    pass\n", names=())

    lines = [f"class {namespace}:"]
    names: list[str] = []
    for spec in specs:
        names.append(spec.name)
        param_doc = _format_param_doc(dict(spec.parameters))
        lines.append("    @staticmethod")
        lines.append(f"    def {spec.name}(**kwargs):")
        lines.append(f'        """{spec.description}\n')
        lines.append(param_doc.rstrip())
        lines.append('        """')
        lines.append("        ...  # runtime-bound")
        lines.append("")
    return ApiStub(source="\n".join(lines), names=tuple(names))
