"""The MCP surface, exercised in-process through the SDK client."""

from __future__ import annotations

from pathlib import Path

import httpx
from mcp import Client

from searxng_jev_mcp.config import Settings
from searxng_jev_mcp.server import App, build_app, create_server

from .conftest import FakeSearx


def make_server(settings: Settings, fake: FakeSearx):
    def factory() -> App:
        return build_app(settings, transport=fake.transport)

    return create_server(app_factory=factory, settings=settings)


async def test_one_tool_with_schema_and_annotations(settings: Settings) -> None:
    async with Client(make_server(settings, FakeSearx())) as client:
        tools = (await client.list_tools()).tools
    assert [t.name for t in tools] == ["search"]
    t = tools[0]
    assert t.annotations and t.annotations.read_only_hint and t.annotations.open_world_hint
    assert t.output_schema and "results" in t.output_schema["properties"]
    assert "untrusted" in (t.description or "")
    props = t.input_schema["properties"]
    for name in ("query", "k", "mode", "categories", "engines", "site", "exclude_domains", "time_range",
                 "language", "queries", "page", "assess", "private", "raw", "explain"):  # fmt: skip
        assert name in props


async def test_search_structured_and_text(settings: Settings) -> None:
    async with Client(make_server(settings, FakeSearx())) as client:
        res = await client.call_tool(
            "search", {"query": "python asyncio gather exceptions", "mode": "fast", "k": 4}
        )
    assert not res.is_error
    sc = res.structured_content
    assert sc and len(sc["results"]) == 4 and sc["ranker"] == "prior"
    text = res.content[0].text  # type: ignore[union-attr]
    assert text.startswith("[1] ") and "coverage:" in text and "untrusted" in text


async def test_explain(settings: Settings) -> None:
    async with Client(make_server(settings, FakeSearx())) as client:
        res = await client.call_tool("search", {"query": "asyncio gather", "mode": "fast", "explain": True})
    sc = res.structured_content
    assert sc and sc["explain"]["timings_s"]["total"] >= 0 and sc["results"][0]["prior_rank"] is not None


async def test_invalid_params_is_error(settings: Settings) -> None:
    async with Client(make_server(settings, FakeSearx())) as client:
        res = await client.call_tool("search", {"query": "x", "categories": ["bogus"], "mode": "fast"})
    assert res.is_error and "Allowed on this instance" in res.content[0].text  # type: ignore[union-attr]


async def test_json_disabled_is_error(settings: Settings) -> None:
    fake = FakeSearx(responder=lambda p: httpx.Response(403))
    async with Client(make_server(settings, fake)) as client:
        res = await client.call_tool("search", {"query": "x y", "mode": "fast"})
    assert res.is_error and "search.formats" in res.content[0].text  # type: ignore[union-attr]


async def test_debug_tools_flag(settings: Settings, tmp_path: Path) -> None:
    s = settings.model_copy(update={"debug_tools": True})
    async with Client(make_server(s, FakeSearx())) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        assert names == ["engines", "jev_status", "route", "search"]
        res = await client.call_tool("jev_status", {})
        assert "set (" not in str(res.structured_content) or "chars" in str(res.structured_content)
