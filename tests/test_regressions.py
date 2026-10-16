"""Regression tests: config loading, provider failover, pipeline failure modes,
time ranges, ordering rules, privacy of on-disk state."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from searxng_jev_mcp import config
from searxng_jev_mcp.decide import providers
from searxng_jev_mcp.decide.backend import ChainDecider, SystemOneDecider
from searxng_jev_mcp.decide.client import DecisionError, SystemOneClient
from searxng_jev_mcp.models import Doc
from searxng_jev_mcp.pipeline import POOL_NS, Pipeline, SearchParams, order_by_decision
from searxng_jev_mcp.searxng import SearxngError
from searxng_jev_mcp.store import TraceLog

from .conftest import FakeSearx, default_responder, load_fixture
from .test_decide import FakeProvider

MakePipeline = Callable[..., Pipeline]


# --------------------------------------------------------------------------------------------- config
def test_working_directory_env_files_are_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "OPENROUTER_BASE_URL=https://attacker.example/api\nSEARXNG_URL=http://evil.example\n"
    )
    (tmp_path / ".env.local").write_text("DECISION_PROVIDER=custom\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "USER_CONFIG_FILE", tmp_path / "missing-user-config")
    s = config.load_settings()
    assert s.searxng_url is None and s.openrouter_base_url == "https://openrouter.ai/api"
    assert s.decision_provider == "auto"


def test_explicit_env_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    a.write_text("SEARXNG_URL=http://first.example\nDECISION_PROVIDER=typesafe\n")
    b.write_text("SEARXNG_URL=http://second.example\n")
    monkeypatch.setattr(config, "USER_CONFIG_FILE", tmp_path / "missing-user-config")
    monkeypatch.setenv("SEARXNG_JEV_ENV_FILE", f"{a}{__import__('os').pathsep}{b}")
    s = config.load_settings(config.default_env_files())
    assert s.searxng_url == "http://second.example" and s.decision_provider == "typesafe"  # later file wins
    monkeypatch.delenv("SEARXNG_JEV_ENV_FILE")
    config.set_cli_env_files([a])
    try:
        assert config.get_settings().searxng_url == "http://first.example"
    finally:
        config.set_cli_env_files([])


def test_keys_never_travel_over_plain_http() -> None:
    assert config.key_transport_ok("https://openrouter.ai/api")
    assert config.key_transport_ok("http://127.0.0.1:11435")
    assert not config.key_transport_ok("http://attacker.example/api")
    with pytest.raises(ValueError, match="refusing to send an API key"):
        SystemOneClient(providers.custom("http://attacker.example", "m", auth=True), "secret")


# --------------------------------------------------------------------------------------------- client / failover
async def test_long_retry_after_is_respected_and_errors_carry_no_provider_text() -> None:
    calls = {"n": 0}

    def handler(body: dict[str, Any]) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "30"},
                              json={"error": {"message": "slow down", "code": 429}, "user_id": "user_secret123"})  # fmt: skip

    c = SystemOneClient(
        providers.openrouter(), "k", transport=httpx.MockTransport(FakeProvider(handler=handler))
    )
    with pytest.raises(DecisionError) as e:
        await c.decide({}, {"a": {}})
    assert calls["n"] == 1 and e.value.kind == "rate_limited"
    assert "user_secret123" not in str(e.value) and "slow down" not in str(e.value)


def _limit_handler(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(403, json={"error": {"message": "Key limit exceeded (monthly limit)", "code": 403}})


async def test_spending_limit_opens_breaker_for_long() -> None:
    c = SystemOneClient(
        providers.openrouter(), "k", transport=httpx.MockTransport(FakeProvider(handler=_limit_handler))
    )
    with pytest.raises(DecisionError) as e:
        await c.decide({}, {"a": {}})
    assert e.value.kind == "quota"
    with pytest.raises(DecisionError) as e2:
        await c.decide({}, {"a": {}})
    assert e2.value.kind == "circuit_open"


def _decider(fake: FakeProvider, profile: providers.ProviderProfile) -> SystemOneDecider:
    return SystemOneDecider(SystemOneClient(profile, "k", transport=httpx.MockTransport(fake)), cache=None)


def _doc(i: int, p: float | None = None, **kw: Any) -> Doc:
    d = Doc(id=f"d{i:03d}", url=f"https://s{i}.example/p", canonical=f"s{i}.example/p", ident=None,
            title=f"t{i}", snippet="s", domain=f"s{i}.example", site_key=f"s{i}.example", prior_rank=i, **kw)  # fmt: skip
    d.p_jev = p
    return d


async def test_chain_fails_over_at_runtime() -> None:
    from searxng_jev_mcp import query

    capped = FakeProvider(handler=_limit_handler)
    backup = FakeProvider(scorer=lambda q, s: 0.7)
    chain = ChainDecider([_decider(capped, providers.openrouter()), _decider(backup, providers.typesafe())])
    docs = [_doc(1), _doc(2)]
    info = await chain.rerank(query.parse_query("x"), docs, navigational=False, deadline=2.0)
    assert info["scored"] == 2 and info["provider"].startswith("typesafe")
    assert chain.name.startswith("typesafe")  # the capped provider stays skipped while its breaker is open
    a = await chain.assess(query.parse_query("x"), docs, None, 1.0)
    assert a.verdict in ("stop", "refine", "broaden", "unsure") and len(capped.bodies) == 1


def test_factory_builds_chain(tmp_path: Path) -> None:
    from searxng_jev_mcp.decide.factory import build_deciders

    s = config.Settings(_env_file=None, searxng_url="http://x", decision_provider="openrouter",  # type: ignore[call-arg]
                        openrouter_api_key="k1", typesafe_api_key="k2", decision_fallback="typesafe",
                        data_dir=tmp_path, keyring_enabled=False)  # fmt: skip
    d, private, reason = build_deciders(s, None)
    assert isinstance(d, ChainDecider) and [x.profile.name for x in d.deciders] == ["openrouter", "typesafe"]
    assert reason is None and private is not None and private.profile.name == "ollaya"
    s2 = s.model_copy(update={"decision_fallback": None})
    d2, _, _ = build_deciders(s2, None)
    assert isinstance(d2, SystemOneDecider)


# --------------------------------------------------------------------------------------------- pipeline failure modes
async def test_exhausted_category_falls_back_instead_of_crashing(
    make_pipeline: MakePipeline, fake_searx: FakeSearx
) -> None:
    p = make_pipeline(fake_searx)
    for _ in range(300):
        p.quota.spend(["stackoverflow", "askubuntu", "superuser"])
    o = await p.search(SearchParams("asyncio gather", mode="fast", categories=["q&a"]))
    codes = {w.code for w in o.warnings}
    assert "quota_skipped" in codes and "categories_unavailable" in codes and o.docs


async def test_no_usable_engines_is_a_clear_error(make_pipeline: MakePipeline) -> None:
    cfg = copy.deepcopy(load_fixture("searxng", "config.json"))
    for e in cfg["engines"]:
        if "general" in e["categories"] and e["name"] != "currency":
            e["enabled"] = False
    with pytest.raises(SearxngError) as e:
        await make_pipeline(FakeSearx(config=cfg)).search(SearchParams("hello there", mode="fast"))
    assert e.value.kind == "no_engines" and "No usable SearXNG engines" in str(e.value)


async def test_total_deadline_with_nothing_back_is_an_error(
    make_pipeline: MakePipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    import searxng_jev_mcp.pipeline as pl

    monkeypatch.setitem(pl.TOTAL_DEADLINE, "fast", 0.6)

    async def slow(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(200, json=load_fixture("searxng", "config.json"))
        await asyncio.sleep(1.5)
        return httpx.Response(200, json=default_responder(dict(request.url.params)))

    fake = FakeSearx()
    fake.handler = slow  # type: ignore[method-assign, assignment]
    with pytest.raises(SearxngError) as e:
        await make_pipeline(fake).search(SearchParams("asyncio gather", mode="fast"))
    assert e.value.kind == "timeout" and "not an empty result" in str(e.value)


async def test_unexpected_exception_becomes_a_clear_tool_error(settings: config.Settings) -> None:
    from mcp import Client

    from searxng_jev_mcp.server import App, build_app, create_server

    def factory() -> App:
        app = build_app(settings, transport=FakeSearx().transport)

        async def boom(params: SearchParams) -> Any:
            raise KeyError("internal")

        app.pipeline.search = boom  # type: ignore[method-assign]
        return app

    async with Client(create_server(app_factory=factory, settings=settings)) as client:
        res = await client.call_tool("search", {"query": "x y", "mode": "fast"})
    assert res.is_error and "Internal error in searxng-jev-mcp (KeyError)" in res.content[0].text  # type: ignore[union-attr]


async def test_time_range_skips_unsupported_engines_visibly(
    make_pipeline: MakePipeline, fake_searx: FakeSearx
) -> None:
    o = await make_pipeline(fake_searx).search(SearchParams("asyncio gather", mode="fast", time_range="week"))
    general_call = fake_searx.calls[0]
    assert general_call["time_range"] == "week"
    assert "wikipedia" not in general_call["engines"] and "bing" in general_call["engines"]
    w = next(w for w in o.warnings if w.code == "time_range_skipped")
    assert "wikipedia" in w.effect


async def test_year_in_query_does_not_restrict_engines(
    make_pipeline: MakePipeline, fake_searx: FakeSearx
) -> None:
    await make_pipeline(fake_searx).search(SearchParams("fastapi lifespan events 2026", mode="fast"))
    assert all("time_range" not in c for c in fake_searx.calls)
    assert "wikipedia" in fake_searx.calls[0]["engines"]


async def test_raw_query_never_written_to_disk(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    p = make_pipeline(fake_searx)
    secret = "sk-test-" + "Ab3" * 12
    params = SearchParams(f"error {secret} in /Users/bob/proj/main.py", mode="fast")
    await p.search(params)
    pooled = p.cache.get(POOL_NS, params.pool_key())
    assert pooled is not None and pooled["meta"]["pq"]["original"] == ""
    dump = json.dumps(pooled)
    assert secret not in dump and "/Users/bob" not in dump


async def test_stale_pool_cache_is_ignored(make_pipeline: MakePipeline, fake_searx: FakeSearx) -> None:
    p = make_pipeline(fake_searx)
    params = SearchParams("asyncio gather", mode="fast", k=3, page=2)
    p.cache.set(POOL_NS, params.pool_key(), {"docs": [{"unexpected": 1}], "meta": {}}, 600)
    o = await p.search(params)
    assert o.docs and fake_searx.calls  # searched again instead of crashing


async def test_routing_off_by_default_and_deciding_when_on(
    make_pipeline: MakePipeline, fake_searx: FakeSearx
) -> None:
    from .test_balanced import ROUTE_P, scorer

    fake = FakeProvider(scorer=scorer(ROUTE_P))
    o = await make_pipeline(fake_searx, decider=_decider(fake, providers.openrouter())).search(
        SearchParams("python asyncio gather exceptions")
    )
    assert o.router == "static" and o.ranker == "jev"
    assert all("documents" in b["state"] for b in fake.bodies)  # rerank only, no routing call
    fake2 = FakeProvider(scorer=scorer({**ROUTE_P, "eng:github": 0.9, "eng:stackoverflow": 0.05}))
    searx2 = FakeSearx()
    o2 = await make_pipeline(
        searx2, decider=_decider(fake2, providers.openrouter()), decision_routing=True
    ).search(SearchParams("python asyncio gather exceptions"))
    vertical = next(c for c in searx2.calls if "github" in c["engines"])
    assert (
        o2.router == "jev" and "stackoverflow" not in vertical["engines"]
    )  # the model decides, static not unioned


async def test_query_variants_do_not_starve_the_vertical_request(make_pipeline: MakePipeline) -> None:
    order: list[str] = []

    def responder(params: dict[str, str]) -> dict[str, Any]:
        order.append("vertical" if "stackoverflow" in params["engines"] else params["q"][:7])
        return default_responder(params)

    fake = FakeSearx(responder=responder)
    await make_pipeline(fake).search(
        SearchParams("python asyncio gather exceptions", mode="fast", queries=["variant one", "variant two"])
    )
    assert order.index("vertical") < order.index("variant")


# --------------------------------------------------------------------------------------------- ordering
def test_near_tie_rule_uses_absolute_gap() -> None:
    docs = [_doc(1, 0.80), _doc(2, 0.815), _doc(3, 0.83)]
    out = order_by_decision(docs, keep=0.2)
    # 0.83 heads the group; 0.815 is within 0.02 → prior decides inside; 0.80 is 0.03 below the head
    assert [d.id for d in out] == ["d002", "d003", "d001"]


def test_authority_guard_keeps_agreed_top_result() -> None:
    official = _doc(5, 0.47, engines={"bing", "yandex", "duckduckgo"}, request_ranks={"general": 1})
    blogs = [_doc(i, 0.9) for i in range(1, 5)]
    out = order_by_decision(
        [*blogs, official],
        keep=0.2,
        guard="pin3",
        general_engines=frozenset({"bing", "yandex", "duckduckgo"}),
    )
    assert out[0].id == "d005"
    # the shipped default (pin3_p80) only protects results the model also rates high
    out_p80 = order_by_decision(
        [*blogs, official], keep=0.2, general_engines=frozenset({"bing", "yandex", "duckduckgo"})
    )
    assert out_p80[0].id != "d005"
    # a model-specific "high" cut (e.g. 0.45 for a model with a compressed scale) protects it again
    out_low = order_by_decision([*blogs, official], keep=0.2, high_p=0.45,
                                general_engines=frozenset({"bing", "yandex", "duckduckgo"}))  # fmt: skip
    assert out_low[0].id == "d005"
    # Bing + Yahoo are one voter: not enough agreement
    same_index = _doc(5, 0.47, engines={"bing", "yahoo"}, request_ranks={"general": 1})
    out2 = order_by_decision(
        [*blogs, same_index], keep=0.2, guard="pin3", general_engines=frozenset({"bing", "yahoo"})
    )
    assert out2[0].id != "d005"
    # a result the model rejects is not protected
    rejected = _doc(5, 0.05, engines={"bing", "yandex"}, request_ranks={"general": 1})
    out3 = order_by_decision(
        [*blogs, rejected], keep=0.2, guard="pin3", general_engines=frozenset({"bing", "yandex"})
    )
    assert out3[-1].id == "d005"


def test_suspects_move_down_a_few_places_not_to_the_end() -> None:
    docs = [_doc(i, 0.9 - i * 0.05) for i in range(1, 12)]
    docs[0].suspect = True
    out = order_by_decision(docs, keep=0.2)
    assert [d.id for d in out].index("d001") == 5 and out[-1].id != "d001"


def test_trace_rotates(tmp_path: Path) -> None:
    t = TraceLog(tmp_path / "trace.jsonl", max_bytes=200)
    for i in range(20):
        t.write({"q": f"query {i}", "pad": "x" * 30})
    assert (tmp_path / "trace.jsonl.1").exists() and (tmp_path / "trace.jsonl").stat().st_size < 400


def test_profiles_carry_local_deadlines() -> None:
    assert providers.openrouter().rerank_deadline_s == 2.0
    local = providers.ollaya()
    assert local.rerank_deadline_s > local.timeout_s / 2 and local.extra_budget_s > 0
    d = _decider(FakeProvider(), dataclasses.replace(local))
    assert d.deadlines[1] == local.rerank_deadline_s and d.keep_threshold == 0.2
