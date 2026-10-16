from __future__ import annotations

import httpx
import pytest

from searxng_jev_mcp.searxng import Catalog, SearxngClient, SearxngError, parse_response

from .conftest import SEARX_URL, FakeSearx, load_fixture


def test_parse_ranks_by_score_not_array_order() -> None:
    body = load_fixture("searxng", "general_python.json")
    resp = parse_response(body, label="general", engines_asked=["bing"], elapsed=0.1)
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in resp.results] == list(range(1, len(resp.results) + 1))
    assert resp.results[0].url.endswith("Python_(programming_language)")
    assert all(r.request == "general" for r in resp.results)


def test_parse_published_and_unresponsive() -> None:
    body = load_fixture("searxng", "general_python.json")
    resp = parse_response(body, label="general", engines_asked=[], elapsed=0.1)
    dated = [r for r in resp.results if r.published]
    assert len(dated) == 1
    assert dated[0].published is not None and dated[0].published.tzinfo is not None
    assert resp.unresponsive == [("duckduckgo", "timeout"), ("yep", "timeout")]
    assert resp.engines_failed == {"duckduckgo", "yep"}


def test_parse_answers_and_infoboxes() -> None:
    resp = parse_response(
        load_fixture("searxng", "answers.json"), label="general", engines_asked=[], elapsed=0
    )
    assert resp.answers == ["100 USD = 86.1 EUR (https://currency.example/)"]
    assert resp.infoboxes[0]["title"] == "Euro"


def test_parse_rejects_non_search_json() -> None:
    with pytest.raises(SearxngError) as e:
        parse_response({"error": "x"}, label="general", engines_asked=[], elapsed=0)
    assert e.value.kind == "bad_response"


def test_parse_tolerates_malformed_rows() -> None:
    body = {
        "results": [{"title": "no url"}, {"url": "https://a.example/", "score": None, "positions": ["x", 2]}]
    }
    resp = parse_response(body, label="general", engines_asked=[], elapsed=0)
    assert len(resp.results) == 1 and resp.results[0].positions == [2] and resp.results[0].score == 0.0


def test_catalog_from_config() -> None:
    cat = Catalog.from_config(load_fixture("searxng", "config.json"))
    assert cat.enabled("bing") and not cat.enabled("google") and not cat.enabled("nonexistent")
    assert "stackoverflow" in cat.in_category("q&a")
    assert "google" not in cat.in_category("general")


async def test_search_sends_timeout_limit_and_engines() -> None:
    fake = FakeSearx()
    client = SearxngClient(SEARX_URL, transport=fake.transport)
    resp = await client.search(
        "q", engines=["bing", "yahoo"], timeout_limit=2.5, language="fa", time_range="week"
    )
    call = fake.calls[0]
    assert call["engines"] == "bing,yahoo" and call["timeout_limit"] == "2.5" and call["format"] == "json"
    assert call["language"] == "fa" and call["time_range"] == "week" and "categories" not in call
    assert resp.engines_asked == ["bing", "yahoo"] and client.live_searches == 1
    await client.aclose()


@pytest.mark.parametrize(("status", "kind"), [(403, "json_disabled"), (429, "rate_limited"), (500, "http")])
async def test_http_errors_are_classified(status: int, kind: str) -> None:
    fake = FakeSearx(responder=lambda p: httpx.Response(status))
    client = SearxngClient(SEARX_URL, transport=fake.transport)
    with pytest.raises(SearxngError) as e:
        await client.search("q", engines=["bing"])
    assert e.value.kind == kind
    await client.aclose()


async def test_unreachable() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = SearxngClient(SEARX_URL, transport=httpx.MockTransport(boom))
    with pytest.raises(SearxngError) as e:
        await client.search("q", engines=["bing"])
    assert e.value.kind == "unreachable"
    await client.aclose()


async def test_redirect_is_soft() -> None:
    fake = FakeSearx(responder=lambda p: httpx.Response(302, headers={"location": "https://example.org/"}))
    client = SearxngClient(SEARX_URL, transport=fake.transport)
    resp = await client.search("!!w test", engines=["bing"])
    assert resp.redirected and resp.results == []
    await client.aclose()


async def test_concurrency_cap() -> None:
    import asyncio

    active = 0
    peak = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return httpx.Response(200, json={"query": "q", "results": []})

    client = SearxngClient(SEARX_URL, transport=httpx.MockTransport(slow_handler), max_concurrency=2)
    await asyncio.gather(*(client.search("q", engines=["bing"]) for _ in range(6)))
    assert peak == 2
    await client.aclose()
