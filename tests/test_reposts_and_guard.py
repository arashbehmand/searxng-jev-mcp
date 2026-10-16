"""Repost dedupe, suggestion filter, authority-guard variants and their explain output."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from searxng_jev_mcp import query
from searxng_jev_mcp.decide import providers
from searxng_jev_mcp.decide.backend import SystemOneDecider
from searxng_jev_mcp.decide.client import SystemOneClient
from searxng_jev_mcp.merge import dedupe_reposts, is_repost, repost_title_key
from searxng_jev_mcp.models import Doc
from searxng_jev_mcp.pipeline import GUARD_MODES, Pipeline, SearchParams, apply_guard, relevant_suggestions
from searxng_jev_mcp.render import build_output

from .conftest import FakeSearx
from .test_decide import FakeProvider

MakePipeline = Callable[..., Pipeline]
ARTICLE = (
    "FastAPI lifespan events replace the deprecated startup and shutdown handlers. Define an async context "
    "manager, yield once the resources are ready, and FastAPI runs the code before the first request and "
    "after the last one."
)


def doc(i: int, url: str, title: str, snippet: str, p: float | None = None, **kw: Any) -> Doc:
    host = url.split("/")[2]
    d = Doc(id=f"d{i:03d}", url=url, canonical=url.split("//")[1], ident=None, title=title, snippet=snippet,
            domain=host, site_key=host, prior_rank=i, **kw)  # fmt: skip
    d.p_jev = p
    return d


# --------------------------------------------------------------------------------------------- reposts
def test_dev_to_repost_is_merged_into_the_better_ranked_copy() -> None:
    original = doc(1, "https://logiclooptech.example/fastapi-lifespan", "FastAPI Lifespan Events: The Complete Guide | LogicLoop Tech",
                   ARTICLE, engines={"bing"})  # fmt: skip
    repost = doc(2, "https://dev.to/someone/fastapi-lifespan-events-the-complete-guide-1abc",
                 "FastAPI Lifespan Events: The Complete Guide - DEV Community", ARTICLE[:150] + " …",
                 engines={"yandex"}, request_ranks={"general": 4})  # fmt: skip
    other = doc(
        3,
        "https://fastapi.example/advanced/events",
        "Lifespan Events - FastAPI",
        "Official docs",
        engines={"bing"},
    )
    out = dedupe_reposts([original, other, repost])
    assert [d.id for d in out] == ["d001", "d003"]
    assert (
        out[0].engines == {"bing", "yandex"}
        and out[0].duplicates == [repost.url]
        and "repost" in out[0].flags
    )
    assert out[0].request_ranks["general"] == 4


def test_version_pages_are_not_merged() -> None:
    a = doc(1, "https://docs.python.example/3.11/whatsnew/3.11.html", "What's New In Python 3.11 — Python documentation",
            "This article explains the new features in Python 3.11, compared to 3.10. Python 3.11 was released on 24 October 2022.")  # fmt: skip
    b = doc(2, "https://mirror.example.org/whatsnew/3.12.html", "What's New In Python 3.12 — Python documentation",
            "This article explains the new features in Python 3.12, compared to 3.11. Python 3.12 was released on 2 October 2023.")  # fmt: skip
    assert not is_repost(a, b)
    assert len(dedupe_reposts([a, b])) == 2


def test_generic_titles_are_not_merged() -> None:
    boiler = "Welcome to the documentation. This guide helps you install the tool and run your first example quickly."
    a = doc(1, "https://alpha.example/docs/start", "Getting Started - Alpha Docs", boiler)
    b = doc(2, "https://beta.example/docs/start", "Getting Started | Beta", boiler)
    assert repost_title_key(a.title) is None and len(dedupe_reposts([a, b])) == 2
    c = doc(3, "https://gamma.example/intro", "Introduction", boiler)
    assert repost_title_key(c.title) is None


def test_same_title_different_articles_are_not_merged() -> None:
    a = doc(1, "https://one.example/decorators", "Understanding Python Decorators in Depth",
            "Decorators wrap a function to extend its behaviour without modifying it, using closures and functools.wraps.")  # fmt: skip
    b = doc(2, "https://two.example/decorators", "Understanding Python Decorators in Depth",
            "In this video course we build a caching layer, a rate limiter and a retry helper with class-based decorators.")  # fmt: skip
    assert not is_repost(a, b)


def test_same_site_copies_are_left_to_url_dedupe() -> None:
    a = doc(1, "https://blog.example/a", "FastAPI Lifespan Events: The Complete Guide", ARTICLE)
    b = doc(2, "https://blog.example/b", "FastAPI Lifespan Events: The Complete Guide", ARTICLE)
    assert len(dedupe_reposts([a, b])) == 2


# --------------------------------------------------------------------------------------------- suggestions
def test_suggestions_must_stay_inside_the_results_topic() -> None:
    # the live case: the engines read "2026" as a date search
    pq = query.parse_query("fastapi lifespan events 2026")
    suggestions = [
        f"fastapi lifespan events 2026 {w}"
        for w in ("schedule", "printable", "calendar printable", "summer", "calendar")
    ]
    pages = [doc(1, "https://fastapi.example/events", "Lifespan Events - FastAPI", ARTICLE),
             doc(2, "https://blog.example/lifespan", "FastAPI lifespan example", "A minimal example of the lifespan context manager.")]  # fmt: skip
    assert relevant_suggestions(pq, suggestions, pages) == []
    assert relevant_suggestions(pq, [*suggestions, "fastapi lifespan events example"], pages) == [
        "fastapi lifespan events example"
    ]


def test_suggestions_need_a_shared_word_and_years_do_not_count() -> None:
    pq = query.parse_query("fastapi lifespan events 2026")
    got = relevant_suggestions(pq, ["printable", "calendar", "summer", "2026 calendar printable",
                                    "fastapi lifespan events example", "lifespan of a cat"])  # fmt: skip
    assert got == ["fastapi lifespan events example", "lifespan of a cat"]
    assert relevant_suggestions(query.parse_query("how to install the 2026"), ["the 2026 holidays"]) == []


# --------------------------------------------------------------------------------------------- guard variants
GE = frozenset({"bing", "yandex", "duckduckgo"})


def ranked_pool() -> list[Doc]:
    homepage = doc(3, "https://fastapi.example/", "FastAPI", "FastAPI framework", 0.53, engines={"bing", "yandex"},
                   request_ranks={"general": 3})  # fmt: skip
    events = doc(1, "https://fastapi.example/advanced/events", "Lifespan Events - FastAPI", "lifespan", 0.72,
                 engines={"bing", "yandex", "duckduckgo"}, request_ranks={"general": 1})  # fmt: skip
    blogs = [
        doc(10 + i, f"https://blog{i}.example/x", f"blog {i}", "s", p)
        for i, p in enumerate((0.83, 0.82, 0.80, 0.79))
    ]
    return sorted([homepage, events, *blogs], key=lambda d: -(d.p_jev or 0))


def test_guard_modes_and_reasons() -> None:
    assert set(GUARD_MODES) >= {
        "off",
        "pin3",
        "pin3_p80",
        "pin3_band",
        "pin1_p80",
        "pin1_band",
        "band_reorder",
    }
    off = apply_guard(ranked_pool(), "off", keep=0.2, general_engines=GE)
    assert [d.p_jev for d in off] == sorted((d.p_jev or 0 for d in off), reverse=True)
    pin3 = apply_guard(ranked_pool(), "pin3", keep=0.2, general_engines=GE)
    assert pin3[0].domain == "fastapi.example" and pin3[2].p_jev == 0.53  # the reported fastapi case
    assert pin3[0].guard_reason and "3 independent" in pin3[0].guard_reason and "guarded" in pin3[0].flags
    p80 = apply_guard(ranked_pool(), "pin3_p80", keep=0.2, general_engines=GE)
    assert [d.p_jev for d in p80][:2] == [0.83, 0.82] and all(d.guard_reason is None for d in p80)
    band = apply_guard(ranked_pool(), "pin1_band", keep=0.2, general_engines=GE)  # 0.72 is 0.11 below best
    assert band[0].p_jev == 0.83
    reorder = apply_guard(ranked_pool(), "band_reorder", keep=0.2, general_engines=GE)
    gaps = [abs((a.p_jev or 0) - (b.p_jev or 0)) for a, b in itertools.pairwise(reorder[:5])]
    assert all(g < 0.2 for g in gaps)
    assert reorder[-1].p_jev == 0.53  # never lifted far above clearly better results


@pytest.mark.parametrize("mode", ["pin3", "off"])
async def test_explain_shows_guard(make_pipeline: MakePipeline, mode: str) -> None:
    def row(url: str, content: str, engines: list[str], score: float) -> dict[str, Any]:
        return {"url": url, "title": url.split("/")[2], "content": content, "engines": engines,
                "positions": [1] * len(engines), "score": score, "category": "general"}  # fmt: skip

    def responder(params: dict[str, str]) -> dict[str, Any]:
        return {"query": params["q"], "results": [
            row("https://docs.example.com/events", "lifespan events reference", ["bing", "yandex"], 4.0),
            *[row(f"https://blog{i}.example/p", "return_exceptions in detail", ["bing"], 1.0 / (i + 2)) for i in range(4)],
        ], "answers": [], "corrections": [], "infoboxes": [], "suggestions": ["printable calendar", "lifespan events reference"],
            "unresponsive_engines": []}  # fmt: skip

    fake = FakeProvider(scorer=lambda q, s: 0.95 if "return_exceptions" in s["documents"][q]["text"] else 0.6)
    d = SystemOneDecider(SystemOneClient(providers.openrouter(), "k", transport=httpx.MockTransport(fake)))
    o = await make_pipeline(FakeSearx(responder=responder), decider=d, guard_mode=mode).search(
        SearchParams("lifespan events reference", explain=True)
    )
    out = build_output(o)
    assert out.suggestions == ["lifespan events reference"]
    top = out.results[0]
    if mode == "pin3":
        assert top.site == "docs.example.com" and top.why and top.why["guarded"] and "guarded" in top.flags
        assert "agree" in top.why["reason"]
    else:
        assert top.site != "docs.example.com" and all(r.why == {"guarded": False} for r in out.results)


def test_unknown_guard_mode_is_rejected(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    with pytest.raises(ValueError, match="unknown authority guard"):
        make_pipeline(fake_searx, guard_mode="sideways")
