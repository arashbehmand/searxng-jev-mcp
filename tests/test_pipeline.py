"""Fast-mode pipeline against synthetic SearXNG fixtures (no network, no decision model)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from searxng_jev_mcp.pipeline import ParamError, Pipeline, SearchParams, age_string, group_unresponsive
from searxng_jev_mcp.render import build_output, render_text
from searxng_jev_mcp.searxng import SearxngError

from .conftest import FakeSearx, load_fixture

MakePipeline = Callable[..., Pipeline]


async def test_fast_mode_basic(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    p = make_pipeline(fake_searx)
    o = await p.search(SearchParams("python asyncio gather exceptions", mode="fast"))
    assert o.ranker == "prior" and o.router == "static"
    assert 1 <= len(o.docs) <= 8
    urls = [d.url for d in o.docs]
    assert len(urls) == len(set(urls))
    # tracking params never shown twice as separate docs; https preferred
    assert all(not u.startswith("http://") or "qa.example.com" in u for u in urls)
    # the static map sends code-ish queries to verticals as request D
    called = fake_searx.engines_called()
    assert any("stackoverflow" in c for c in called)
    general_call = next(c for c in called if "bing" in c)
    assert "google" not in general_call and "brave" not in general_call  # disabled engines never named
    assert {d.request_ranks.keys().__iter__().__next__() for d in o.docs} <= {"general", "vertical"}


async def test_grouped_timeout_warning(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    o = await make_pipeline(fake_searx).search(SearchParams("python asyncio gather exceptions", mode="fast"))
    timeouts = [w for w in o.warnings if w.code == "engines_timeout"]
    assert len(timeouts) == 1 and "duckduckgo, yep" in timeouts[0].effect
    assert not o.degraded  # 2 of 9 engines timing out is a warning, not a degraded search
    out = build_output(o)
    assert out.coverage.engines_asked >= out.coverage.engines_ok


async def test_all_engines_failed_is_a_failure_not_empty(make_pipeline: MakePipeline) -> None:
    fake = FakeSearx(
        responder=lambda p: {
            **load_fixture("searxng", "all_failed.json"),
            "unresponsive_engines": [[e, "timeout"] for e in p["engines"].split(",")],
        }
    )
    o = await make_pipeline(fake).search(SearchParams("kubernetes pod crashloopbackoff", mode="fast"))
    assert o.docs == [] and o.degraded
    assert any(w.code == "all_engines_failed" for w in o.warnings)
    text = render_text(build_output(o))
    assert "FAILED" in text and "No results." not in text


async def test_empty_is_not_a_failure(make_pipeline: MakePipeline) -> None:
    fake = FakeSearx(responder=lambda p: load_fixture("searxng", "empty.json"))
    o = await make_pipeline(fake).search(SearchParams("zxqvbnm qwertyuiop", mode="fast"))
    assert o.docs == [] and not any(w.code == "all_engines_failed" for w in o.warnings)
    assert render_text(build_output(o)).startswith("No results.")


async def test_searxng_syntax_is_escaped(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    await make_pipeline(fake_searx).search(SearchParams("pandas <2 !wp :fr", mode="fast"))
    q = fake_searx.calls[0]["q"]
    assert "<2" not in q and "!wp" not in q and ":fr" not in q
    assert "< 2" in q and "! wp" in q and ": fr" in q


async def test_raw_passes_syntax_through(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    await make_pipeline(fake_searx).search(SearchParams("pandas <2", mode="fast", raw=True))
    assert "pandas <2" in fake_searx.calls[0]["q"]


async def test_site_param_filters_and_forwards(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    o = await make_pipeline(fake_searx).search(
        SearchParams("asyncio gather", mode="fast", site="docs.example.com")
    )
    assert "site:docs.example.com" in fake_searx.calls[0]["q"]
    assert o.docs and all(d.domain.endswith("docs.example.com") for d in o.docs)


async def test_exclude_domains(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    o = await make_pipeline(fake_searx).search(
        SearchParams("asyncio gather", mode="fast", exclude_domains=["wikipedia.org", "python.org"])
    )
    assert o.docs and not any("wikipedia.org" in d.url or "python.org/" in d.url for d in o.docs)


async def test_cache_and_paging(make_pipeline: MakePipeline) -> None:
    from .conftest import default_responder

    # no timeouts, so the computed baseline (and the cache key) stays the same between calls
    fake_searx = FakeSearx(responder=lambda prm: {**default_responder(prm), "unresponsive_engines": []})
    p = make_pipeline(fake_searx)
    o1 = await p.search(SearchParams("python asyncio gather exceptions", mode="fast", k=3))
    n_calls = len(fake_searx.calls)
    o2 = await p.search(SearchParams("python asyncio gather exceptions", mode="fast", k=3, page=2))
    assert len(fake_searx.calls) == n_calls  # page 2 comes from the ranked pool
    assert not {d.url for d in o1.docs} & {d.url for d in o2.docs}
    assert o1.has_more
    out = build_output(o2)
    assert out.results[0].n == 4  # anchors continue across pages
    await p.search(SearchParams("python asyncio gather exceptions", mode="fast", k=5))
    assert len(fake_searx.calls) == n_calls  # SearXNG responses are cached


async def test_caller_engines(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    p = make_pipeline(fake_searx)
    o = await p.search(SearchParams("asyncio", mode="fast", engines=["bing", "google"]))
    assert fake_searx.engines_called() == [["bing"]]
    assert o.router == "caller" and any(w.code == "engines_refused" for w in o.warnings)
    with pytest.raises(ParamError, match="Enabled examples"):
        await p.search(SearchParams("asyncio", mode="fast", engines=["google", "brave"]))


async def test_caller_categories(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    await make_pipeline(fake_searx).search(SearchParams("asyncio gather", mode="fast", categories=["q&a"]))
    assert fake_searx.engines_called() == [["stackoverflow", "askubuntu", "superuser"]]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"query": "  "}, "non-empty"),
        ({"query": "x", "k": 0}, "between 1 and 20"),
        ({"query": "x", "time_range": "decade"}, "day, week, month, year"),
        ({"query": "x", "categories": ["nope"]}, "Allowed on this instance"),
        ({"query": "x", "queries": ["a", "b", "c", "d"]}, "at most 3"),
    ],
)
async def test_invalid_params(
    make_pipeline: MakePipeline, fake_searx: FakeSearx, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ParamError, match=match):
        await make_pipeline(fake_searx).search(SearchParams(mode="fast", **kwargs))


async def test_unreachable_raises(make_pipeline: MakePipeline) -> None:
    def responder(p: dict[str, str]) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(SearxngError) as e:
        await make_pipeline(FakeSearx(responder=responder)).search(SearchParams("x y", mode="fast"))
    assert e.value.kind == "unreachable"


async def test_answers_and_infobox(make_pipeline: MakePipeline) -> None:
    fake = FakeSearx(responder=lambda p: load_fixture("searxng", "answers.json"))
    o = await make_pipeline(fake).search(SearchParams("100 usd to eur", mode="fast"))
    text = render_text(build_output(o))
    assert "86.1 EUR" in text and "infobox: Euro" in text


async def test_health_feeds_baseline(make_pipeline: MakePipeline) -> None:
    def responder(p: dict[str, str]) -> dict[str, Any]:
        body = load_fixture("searxng", "general_python.json")
        body["unresponsive_engines"] = [["yandex", "Suspended: CAPTCHA"]]
        return body

    fake = FakeSearx(responder=responder)
    p = make_pipeline(fake)
    await p.search(SearchParams("first query", mode="fast"))
    await p.search(SearchParams("second query", mode="fast"))
    assert "yandex" in fake.engines_called()[0] and "yandex" not in fake.engines_called()[-1]


async def test_balanced_without_decider_degrades(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    o = await make_pipeline(fake_searx).search(SearchParams("python asyncio gather exceptions"))
    assert o.ranker == "prior" and any(w.code == "decisions_unavailable" for w in o.warnings)
    out = build_output(o)
    assert all(r.rel is None for r in out.results)


async def test_variants_fused(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    o = await make_pipeline(fake_searx).search(
        SearchParams("asyncio gather", mode="fast", queries=["gather return_exceptions"])
    )
    assert any(c["q"].startswith("gather return_exceptions") for c in fake_searx.calls)
    assert o.docs


async def test_scrub_warning_and_trace(
    make_pipeline: MakePipeline, fake_searx: FakeSearx, tmp_path: Any
) -> None:
    o = await make_pipeline(fake_searx).search(
        SearchParams("error in /home/alice/project/src/main.py line 3", mode="fast")
    )
    assert "/home/alice" not in fake_searx.calls[0]["q"]
    assert any(w.code == "query_scrubbed" for w in o.warnings)
    trace = (tmp_path / "trace.jsonl").read_text()
    assert "alice" not in trace


def test_age_string() -> None:
    import datetime as dt

    now = dt.datetime(2026, 10, 8, tzinfo=dt.UTC)
    assert age_string(None) is None
    assert age_string(now - dt.timedelta(hours=3), now) == "today"
    assert age_string(now - dt.timedelta(days=3), now) == "3 days"
    assert age_string(now - dt.timedelta(days=240), now) == "8 months"
    assert age_string(now - dt.timedelta(days=800), now) == "2 years"
    assert age_string(dt.datetime(2026, 10, 1), now) == "7 days"


def test_group_unresponsive() -> None:
    w = group_unresponsive([("yep", "timeout"), ("duckduckgo", "timeout"), ("yandex", "Suspended: CAPTCHA")])
    assert [x.code for x in w] == ["engines_captcha", "engines_timeout"]
    assert "duckduckgo, yep" in w[1].effect


def test_static_routing_needs_tech_context_for_qa() -> None:
    from searxng_jev_mcp import query, route

    assert "q&a" not in route.static_categories(query.parse_query("how to make a sourdough starter"))
    cats = route.static_categories(query.parse_query("how to fix TypeError in python pandas"))
    assert "it" in cats and "q&a" in cats
