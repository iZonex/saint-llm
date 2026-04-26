"""Parse tool-call structures out of generated LM text.

A trained chat LM that knows how to use tools emits structured
markers in its output that the runtime interprets as tool calls.
Two conventions are common:

* **XML-tag style** (Anthropic Claude / Hermes / Mistral)::

      <tool_call name="search" id="call_1">
      {"query": "weather"}
      </tool_call>

* **JSON-fenced style** (OpenAI / DeepSeek / Qwen)::

      ```json
      {"name": "search", "arguments": {"query": "weather"}}
      ```

Both formats embed a tool name and an arguments object. This module
provides parsers that extract them as :class:`ToolCall` objects ready
for :meth:`ToolRegistry.execute`. Multiple tool calls in one assistant
turn are supported — the parser returns them in source order.

Malformed blocks (missing ``name``, invalid JSON, mismatched tags)
are silently skipped — the model output may contain partial / garbled
attempts mixed with valid ones, and dropping the bad entries is
preferable to raising on the whole turn. Callers can opt into strict
mode via ``strict=True`` to raise on any malformed block.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Literal

from saint_llm_agents.message import ToolCall

ParserFormat = Literal["xml", "json", "auto"]

# XML-style: <tool_call name="..." [id="..."]>...JSON body...</tool_call>
_XML_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+([^>]+)>(.*?)</tool_call>',
    re.DOTALL,
)
_XML_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# JSON-fenced: ```json\n{...}\n``` or ```\n{...}\n```
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.+?)\n```",
    re.DOTALL,
)


def parse_tool_calls(
    text: str,
    *,
    fmt: ParserFormat = "auto",
    strict: bool = False,
) -> list[ToolCall]:
    """Extract :class:`ToolCall` objects from ``text``.

    Args:
        text:    LM output to parse.
        fmt:     ``"xml"`` to parse only XML tags, ``"json"`` only
            JSON fences, ``"auto"`` (default) tries both and returns
            the union (XML hits first, then JSON, deduplicated by
            content).
        strict:  if True, malformed blocks raise :class:`ValueError`.
            Default False — silently skip bad blocks.

    Returns:
        list of :class:`ToolCall` in source order. Empty list when no
        tool calls are detected.
    """
    if fmt == "xml":
        return _parse_xml(text, strict=strict)
    if fmt == "json":
        return _parse_json(text, strict=strict)
    if fmt == "auto":
        xml_calls = _parse_xml(text, strict=strict)
        json_calls = _parse_json(text, strict=strict)
        # Deduplicate JSON calls that overlap with XML calls (heuristic:
        # same name + same arguments). XML wins ties since it carries
        # explicit IDs.
        seen = {(c.name, _hashable_args(c.arguments)) for c in xml_calls}
        out = list(xml_calls)
        for c in json_calls:
            key = (c.name, _hashable_args(c.arguments))
            if key not in seen:
                out.append(c)
                seen.add(key)
        return out
    raise ValueError(f"unknown parser format {fmt!r}")


def _parse_xml(text: str, *, strict: bool) -> list[ToolCall]:
    out: list[ToolCall] = []
    for match in _XML_TOOL_CALL_RE.finditer(text):
        attrs_text, body = match.group(1), match.group(2).strip()
        attrs = dict(_XML_ATTR_RE.findall(attrs_text))
        name = attrs.get("name")
        if not name:
            if strict:
                raise ValueError(f"tool_call tag missing name: {match.group(0)!r}")
            continue
        try:
            arguments = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(
                    f"tool_call {name!r} has invalid JSON body: {body!r}",
                ) from exc
            continue
        if not isinstance(arguments, dict):
            if strict:
                raise ValueError(
                    f"tool_call {name!r} arguments must be a JSON object; "
                    f"got {type(arguments).__name__}",
                )
            continue
        call_id = attrs.get("id") or _gen_id()
        out.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return out


def _parse_json(text: str, *, strict: bool) -> list[ToolCall]:
    out: list[ToolCall] = []
    for match in _JSON_FENCE_RE.finditer(text):
        body = match.group(1).strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"json-fenced block has invalid JSON: {body!r}") from exc
            continue
        # Accept both a single object and a list of objects.
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                if strict:
                    raise ValueError(
                        f"json-fenced item must be an object; got {type(item).__name__}",
                    )
                continue
            name = item.get("name")
            if not name or not isinstance(name, str):
                if strict:
                    raise ValueError(f"json-fenced item missing 'name': {item!r}")
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                if strict:
                    raise ValueError(
                        f"json-fenced item 'arguments' must be an object; "
                        f"got {type(arguments).__name__}",
                    )
                continue
            call_id = item.get("id") or _gen_id()
            if not isinstance(call_id, str):
                call_id = str(call_id)
            out.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return out


def _gen_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _hashable_args(args: object) -> str:
    """JSON-serialize args for dedup-equality checks across parsers."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        return str(args)
