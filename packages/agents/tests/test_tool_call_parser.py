"""Tests for the tool-call text parser."""

from __future__ import annotations

import pytest
from saint_llm_agents import ToolCall
from saint_llm_agents.tool_call_parser import parse_tool_calls

# ---- XML format -----------------------------------------------------


def test_xml_single_call_basic() -> None:
    text = (
        '<tool_call name="search" id="call_1">\n'
        '{"query": "weather"}\n'
        '</tool_call>'
    )
    calls = parse_tool_calls(text, fmt="xml")
    assert len(calls) == 1
    assert calls[0] == ToolCall(
        id="call_1", name="search", arguments={"query": "weather"},
    )


def test_xml_assigns_id_when_missing() -> None:
    text = '<tool_call name="ping">{}</tool_call>'
    calls = parse_tool_calls(text, fmt="xml")
    assert len(calls) == 1
    assert calls[0].name == "ping"
    assert calls[0].id.startswith("call_")


def test_xml_empty_body_yields_empty_args() -> None:
    text = '<tool_call name="now"></tool_call>'
    calls = parse_tool_calls(text, fmt="xml")
    assert calls[0].arguments == {}


def test_xml_multiple_calls_in_order() -> None:
    text = (
        '<tool_call name="a" id="1">{"x": 1}</tool_call>'
        '<tool_call name="b" id="2">{"y": 2}</tool_call>'
    )
    calls = parse_tool_calls(text, fmt="xml")
    assert [c.name for c in calls] == ["a", "b"]


def test_xml_skips_malformed_in_lenient_mode() -> None:
    text = (
        '<tool_call name="ok" id="1">{"x": 1}</tool_call>'
        '<tool_call name="bad" id="2">not-json</tool_call>'
    )
    calls = parse_tool_calls(text, fmt="xml")
    assert len(calls) == 1
    assert calls[0].name == "ok"


def test_xml_strict_raises_on_invalid_json() -> None:
    text = '<tool_call name="bad">not-json</tool_call>'
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_tool_calls(text, fmt="xml", strict=True)


def test_xml_strict_raises_on_missing_name() -> None:
    text = '<tool_call id="x">{"a": 1}</tool_call>'
    with pytest.raises(ValueError, match="missing name"):
        parse_tool_calls(text, fmt="xml", strict=True)


def test_xml_skips_non_object_args_in_lenient_mode() -> None:
    text = '<tool_call name="bad" id="1">[1, 2, 3]</tool_call>'
    calls = parse_tool_calls(text, fmt="xml")
    assert calls == []


def test_xml_strict_raises_on_non_object_args() -> None:
    text = '<tool_call name="bad" id="1">[1, 2, 3]</tool_call>'
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_tool_calls(text, fmt="xml", strict=True)


def test_xml_handles_text_around_calls() -> None:
    text = (
        "Sure! Let me look that up.\n"
        '<tool_call name="search" id="c1">{"query": "x"}</tool_call>\n'
        "Hold on..."
    )
    calls = parse_tool_calls(text, fmt="xml")
    assert len(calls) == 1
    assert calls[0].name == "search"


def test_xml_no_calls_returns_empty() -> None:
    assert parse_tool_calls("just some text", fmt="xml") == []


# ---- JSON-fenced format ---------------------------------------------


def test_json_fenced_basic() -> None:
    text = (
        "Calling tool:\n"
        "```json\n"
        '{"name": "search", "arguments": {"query": "weather"}}\n'
        "```"
    )
    calls = parse_tool_calls(text, fmt="json")
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "weather"}


def test_json_fenced_without_lang_tag() -> None:
    text = '```\n{"name": "x", "arguments": {}}\n```'
    calls = parse_tool_calls(text, fmt="json")
    assert calls[0].name == "x"


def test_json_fenced_list_yields_multiple_calls() -> None:
    text = (
        "```json\n"
        '[{"name": "a", "arguments": {"x": 1}}, '
        '{"name": "b", "arguments": {"y": 2}}]\n'
        "```"
    )
    calls = parse_tool_calls(text, fmt="json")
    assert [c.name for c in calls] == ["a", "b"]


def test_json_fenced_caller_id_preserved() -> None:
    text = (
        '```json\n{"id": "my_id", "name": "x", "arguments": {}}\n```'
    )
    calls = parse_tool_calls(text, fmt="json")
    assert calls[0].id == "my_id"


def test_json_fenced_skips_missing_name_in_lenient_mode() -> None:
    text = '```json\n{"arguments": {"x": 1}}\n```'
    calls = parse_tool_calls(text, fmt="json")
    assert calls == []


def test_json_fenced_strict_raises_on_missing_name() -> None:
    text = '```json\n{"arguments": {"x": 1}}\n```'
    with pytest.raises(ValueError, match="missing 'name'"):
        parse_tool_calls(text, fmt="json", strict=True)


def test_json_fenced_skips_non_object_arguments_in_lenient_mode() -> None:
    text = '```json\n{"name": "x", "arguments": [1, 2, 3]}\n```'
    calls = parse_tool_calls(text, fmt="json")
    assert calls == []


def test_json_fenced_strict_raises_on_invalid_json() -> None:
    text = "```json\nnot json\n```"
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_tool_calls(text, fmt="json", strict=True)


def test_json_fenced_default_empty_arguments() -> None:
    text = '```json\n{"name": "noargs"}\n```'
    calls = parse_tool_calls(text, fmt="json")
    assert calls[0].arguments == {}


# ---- auto format ----------------------------------------------------


def test_auto_returns_xml_calls_first_then_json() -> None:
    text = (
        '<tool_call name="from_xml" id="x1">{"a": 1}</tool_call>\n'
        '```json\n{"name": "from_json", "arguments": {"b": 2}}\n```'
    )
    calls = parse_tool_calls(text, fmt="auto")
    names = [c.name for c in calls]
    assert names == ["from_xml", "from_json"]


def test_auto_dedups_when_same_call_in_both_formats() -> None:
    """If model emits same call in both XML and JSON, dedup keeps XML version."""
    text = (
        '<tool_call name="x" id="from_xml">{"a": 1}</tool_call>\n'
        '```json\n{"name": "x", "arguments": {"a": 1}}\n```'
    )
    calls = parse_tool_calls(text, fmt="auto")
    assert len(calls) == 1
    assert calls[0].id == "from_xml"


def test_auto_keeps_json_calls_with_distinct_args() -> None:
    text = (
        '<tool_call name="x" id="x1">{"a": 1}</tool_call>\n'
        '```json\n{"name": "x", "arguments": {"a": 2}}\n```'
    )
    calls = parse_tool_calls(text, fmt="auto")
    assert len(calls) == 2


def test_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="unknown parser format"):
        parse_tool_calls("x", fmt="bogus")  # type: ignore[arg-type]


def test_empty_text_yields_no_calls() -> None:
    assert parse_tool_calls("", fmt="auto") == []


def test_parsed_calls_are_tool_call_dataclass() -> None:
    text = '<tool_call name="x" id="1">{}</tool_call>'
    [call] = parse_tool_calls(text, fmt="xml")
    assert isinstance(call, ToolCall)
    assert call.name == "x"
