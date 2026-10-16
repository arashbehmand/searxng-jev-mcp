"""Engine selection without a decision model: baseline, category engines, static rules, quota."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from searxng_jev_mcp import route
from searxng_jev_mcp.health import HealthTable
from searxng_jev_mcp.models import ParsedQuery
from searxng_jev_mcp.route import QuotaBuckets, RoutingCatalog, load_catalog
from searxng_jev_mcp.searxng import Catalog
from searxng_jev_mcp.store import Cache, Store

NOW = 1_000_000.0
CONFIG_PATH = Path(__file__).parent / "fixtures" / "searxng" / "config.json"
CORE = ["bing", "yahoo", "yandex", "wikipedia"]


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def pq(text: str) -> ParsedQuery:
    return ParsedQuery(original=text, text=text, lexical_text=text)


def open_circuits(health: HealthTable, engines: list[str], now: float = NOW) -> None:
    for e in engines:
        health.record([e], [(e, "Suspended: CAPTCHA")], 1.0, now=now)


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.from_config(config())


@pytest.fixture
def health() -> HealthTable:
    return HealthTable(Store(":memory:"))


@pytest.fixture
def rc() -> RoutingCatalog:
    return load_catalog()


# --------------------------------------------------------------------------------------------- baseline
def test_baseline_uses_core_conditional_and_flag(
    catalog: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    engines, probing = route.general_baseline_with_status(catalog, health, rc, now=NOW)
    assert engines[:4] == CORE and not probing
    assert "duckduckgo" in engines and "yep" in engines  # unknown health: half-open probe
    assert "google cse" in engines and "currency" not in engines and "google" not in engines
    no_cse = route.general_baseline(catalog, health, rc, google_cse=False, now=NOW)
    assert "google cse" not in no_cse


def test_baseline_drops_open_core_engines_while_others_work(
    catalog: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    open_circuits(health, ["bing"])
    engines, probing = route.general_baseline_with_status(catalog, health, rc, now=NOW + 60)
    assert "bing" not in engines and {"yahoo", "yandex", "wikipedia"} <= set(engines)
    assert not probing


def test_baseline_probes_core_when_all_core_open(
    catalog: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    open_circuits(health, [*CORE, "duckduckgo", "yep"])
    engines, probing = route.general_baseline_with_status(catalog, health, rc, google_cse=False, now=NOW + 60)
    assert engines == CORE and probing
    assert route.general_baseline(catalog, health, rc, google_cse=False, now=NOW + 60) == CORE
    # with Google CSE healthy the probe still includes the core engines
    engines, probing = route.general_baseline_with_status(catalog, health, rc, now=NOW + 60)
    assert engines == [*CORE, "google cse"] and probing
    # after the cool-down the circuits are closed again: no probe flag
    _, probing = route.general_baseline_with_status(catalog, health, rc, now=NOW + 25 * 3600)
    assert not probing


def test_baseline_probe_skips_engines_forced_off(
    catalog: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    open_circuits(health, CORE)
    health.overrides = {"bing": False}
    engines, probing = route.general_baseline_with_status(catalog, health, rc, google_cse=False, now=NOW + 60)
    assert probing and "bing" not in engines and engines[:3] == ["yahoo", "yandex", "wikipedia"]


def test_baseline_unknown_layout_never_empty(
    catalog: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    bare = dataclasses.replace(rc, general={"flagged": ["google cse"]})  # no core/conditional lists
    engines, probing = route.general_baseline_with_status(catalog, health, bare, google_cse=False, now=NOW)
    assert not probing and "bing" in engines
    assert "currency" not in engines and "wikidata" not in engines and "google cse" not in engines
    open_circuits(health, engines)
    probed, probing = route.general_baseline_with_status(
        catalog, health, bare, google_cse=False, now=NOW + 60
    )
    assert probing and probed == engines


# --------------------------------------------------------------------------------------------- categories
@pytest.fixture
def catalog_with_unlisted() -> Catalog:
    cfg = copy.deepcopy(config())
    cfg["engines"] += [
        {"name": "qwant news", "categories": ["news"], "enabled": True},
        {"name": "yep news", "categories": ["news"], "enabled": False},
        {"name": "codeberg", "categories": ["it", "repos"], "enabled": True},
    ]
    return Catalog.from_config(cfg)


def test_include_unlisted_off_keeps_catalog_list(
    catalog_with_unlisted: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    got = route.engines_for_categories(["news"], pq("x"), catalog_with_unlisted, health, rc, gated=False)
    assert got == ["bing news", "wikinews"]


def test_include_unlisted_adds_unknown_instance_engines(
    catalog_with_unlisted: Catalog, health: HealthTable, rc: RoutingCatalog
) -> None:
    def pick(cats: list[str], **kw: Any) -> list[str]:
        return route.engines_for_categories(
            cats,
            pq("x"),
            catalog_with_unlisted,
            health,
            rc,
            gated=False,
            include_unlisted=True,
            now=NOW + 60,
            **kw,
        )

    assert pick(["news"]) == ["bing news", "wikinews", "qwant news"]  # disabled "yep news" never
    assert pick(["news"], exclude={"qwant news"}) == ["bing news", "wikinews"]
    assert pick(["news"], cap=2) == ["bing news", "wikinews"]
    # engines the routing catalog knows (pypi, docker hub, askubuntu…) are not pulled in by "it"
    assert pick(["it"]) == ["stackoverflow", "github", "mdn", "arch linux wiki", "codeberg"]
    open_circuits(health, ["qwant news"])
    assert "qwant news" not in pick(["news"])


def test_unlisted_category_uses_instance_engines(health: HealthTable, rc: RoutingCatalog) -> None:
    cfg = copy.deepcopy(config())
    cfg["engines"].append({"name": "flickr", "categories": ["images"], "enabled": True})
    cat = Catalog.from_config(cfg)
    assert route.engines_for_categories(["images"], pq("x"), cat, health, rc) == ["flickr"]


def test_engine_gates() -> None:
    assert route.engine_gate_ok("mdn", pq("css grid auto-fit"))
    assert not route.engine_gate_ok("mdn", pq("python asyncio gather"))
    assert route.engine_gate_ok("arch linux wiki", pq("systemd boot hangs"))
    assert not route.engine_gate_ok("hoogle", pq("python list comprehension"))
    assert route.engine_gate_ok("stackoverflow", pq("anything at all"))  # no gate


def test_gated_category_engines(catalog: Catalog, health: HealthTable, rc: RoutingCatalog) -> None:
    q = pq("python asyncio gather")
    gated = route.engines_for_categories(["it"], q, catalog, health, rc)
    assert gated == ["stackoverflow", "github"]
    ungated = route.engines_for_categories(["it"], q, catalog, health, rc, gated=False)
    assert ungated == ["stackoverflow", "github", "mdn", "arch linux wiki"]


# --------------------------------------------------------------------------------------------- static map
def test_static_qa_needs_tech_context() -> None:
    cats = route.static_categories(pq("how to make a sourdough starter"))
    assert "q&a" not in cats and "it" not in cats
    cats = route.static_categories(pq("how to fix TypeError in python pandas"))
    assert "it" in cats and "q&a" in cats


def test_static_categories_basics() -> None:
    assert route.static_categories(pq("arxiv paper on diffusion models"))[:1] == ["science"]
    assert "dictionaries" in route.static_categories(pq("etymology of serendipity"))
    # a code token alone implies it + q&a
    assert route.static_categories(pq("parse_config returns None")) == ["it", "q&a"]
    assert route.static_categories(pq("best hiking trails near the lake")) == []


# --------------------------------------------------------------------------------------------- quota
@pytest.fixture
def quota(rc: RoutingCatalog) -> QuotaBuckets:
    return QuotaBuckets(Cache(Store(":memory:")), rc)


SE = ["stackoverflow", "askubuntu", "superuser"]


def test_quota_counts_every_engine(quota: QuotaBuckets) -> None:
    for _ in range(83):  # 83 requests x 3 Stack Exchange engines = 249 API calls
        quota.spend(SE, now=NOW)
    assert quota.allow("stackoverflow", now=NOW)
    assert quota.filter([*SE, "bing"], now=NOW) == (["stackoverflow", "bing"], ["askubuntu", "superuser"])
    quota.spend(["stackoverflow"], now=NOW)  # 250
    assert not quota.allow("superuser", now=NOW)
    assert quota.filter(SE, now=NOW) == ([], SE)
    assert quota.filter(SE, now=NOW + 86400) == (SE, [])  # next window


def test_quota_filter_and_duplicates(rc: RoutingCatalog) -> None:
    quota = QuotaBuckets(Cache(Store(":memory:"), enabled=False), rc)  # local counters only
    quota.spend(["github", "github", "bing"], now=NOW)  # one call, not two
    allowed, skipped = quota.filter(["github", "github", "mdn"], now=NOW)
    assert allowed == ["github", "mdn"] and skipped == []
    for _ in range(7):
        quota.spend(["github"], now=NOW)
    assert quota.filter(["mdn", "github"], now=NOW) == (["mdn"], ["github"])  # 8/min used
    assert quota.allow("github", now=NOW + 60)
