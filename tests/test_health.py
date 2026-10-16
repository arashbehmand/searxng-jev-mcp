from __future__ import annotations

import pytest

from searxng_jev_mcp.health import (
    HOUR_OPEN,
    LONG_OPEN,
    SHORT_OPEN,
    EngineHealth,
    HealthTable,
    classify_reason,
    parse_overrides,
)
from searxng_jev_mcp.store import Cache, Store

NOW = 1_000_000.0


@pytest.fixture
def table() -> HealthTable:
    return HealthTable(Store(":memory:"))


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("timeout", "timeout"),
        ("read timed out", "timeout"),
        ("Suspended: CAPTCHA", "captcha"),
        ("CAPTCHA", "captcha"),
        ("Suspended: access denied", "blocked"),
        ("HTTP error 403", "blocked"),
        ("Forbidden", "blocked"),
        ("Suspended: too many requests", "rate_limited"),
        ("too many requests", "rate_limited"),
        ("HTTP error 429", "rate_limited"),
        ("API rate limit exceeded", "rate_limited"),
        ("Suspended: daily quota exceeded", "rate_limited"),
        ("Suspended: timeout", "suspended"),
        ("Suspended: unexpected crash", "suspended"),
        ("unexpected crash", "error"),
        ("HTTP error", "error"),
    ],
)
def test_classify_reason(reason: str, kind: str) -> None:
    assert classify_reason(reason) == kind


def test_classify_reason_precedence() -> None:
    # CAPTCHA beats everything; rate-limit words beat access-denied words (GitHub's 403 rate limit)
    assert classify_reason("Suspended: access denied (CAPTCHA)") == "captcha"
    assert classify_reason("403 Forbidden: API rate limit exceeded") == "rate_limited"


def test_two_failures_open_circuit(table: HealthTable) -> None:
    table.record(["yep"], [("yep", "timeout")], 2.5, now=NOW)
    assert table.usable("yep", now=NOW + 1)
    table.record(["yep"], [("yep", "timeout")], 2.5, now=NOW + 2)
    assert not table.usable("yep", now=NOW + 3)
    assert table.usable("yep", now=NOW + 2 + SHORT_OPEN + 1)  # half-open after 15 min


def test_two_errors_open_circuit(table: HealthTable) -> None:
    table.record(["yep"], [("yep", "HTTP error")], 0.5, now=NOW)
    assert table.usable("yep", now=NOW + 1)
    table.record(["yep"], [("yep", "parsing error")], 0.5, now=NOW + 2)
    assert not table.usable("yep", now=NOW + 3)
    assert table.usable("yep", now=NOW + 2 + SHORT_OPEN + 1)


def test_captcha_opens_for_a_day(table: HealthTable) -> None:
    table.record(["duckduckgo"], [("duckduckgo", "Suspended: CAPTCHA")], 1.0, now=NOW)
    assert table.get("duckduckgo").last_reason == "captcha"
    assert not table.usable("duckduckgo", now=NOW + 23 * 3600)
    assert table.usable("duckduckgo", now=NOW + LONG_OPEN + 1)


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("Suspended: access denied", "blocked"),  # e.g. GitHub's API rate limit arrives as HTTP 403
        ("Suspended: too many requests", "rate_limited"),
        ("Suspended: unexpected crash", "suspended"),
    ],
)
def test_transient_blocks_open_for_an_hour(table: HealthTable, reason: str, kind: str) -> None:
    table.record(["github"], [("github", reason)], 1.0, now=NOW)
    assert table.get("github").last_reason == kind
    assert not table.usable("github", now=NOW + HOUR_OPEN - 60)
    assert table.usable("github", now=NOW + HOUR_OPEN + 1)


def test_weaker_failure_does_not_shorten_captcha(table: HealthTable) -> None:
    table.record(["duckduckgo"], [("duckduckgo", "Suspended: CAPTCHA")], 1.0, now=NOW)
    # explicitly asked again while open; two timeouts would normally mean 15 minutes
    table.record(["duckduckgo"], [("duckduckgo", "timeout")], 2.5, now=NOW + 10)
    table.record(["duckduckgo"], [("duckduckgo", "timeout")], 2.5, now=NOW + 20)
    assert not table.usable("duckduckgo", now=NOW + 3600)


def test_instance_wide_timeout_opens_nothing(table: HealthTable) -> None:
    asked = ["bing", "yahoo", "yandex"]
    for i in range(3):
        table.record(asked, [(e, "timeout") for e in asked], 3.0, now=NOW + i)
    for e in asked:
        h = table.get(e)
        assert table.usable(e, now=NOW + 5)
        assert h.n == 0 and h.fail_streak == 0 and h.last_reason is None
        assert h.last_probe == NOW + 2


def test_partial_timeout_still_penalised(table: HealthTable) -> None:
    for i in range(2):
        table.record(["bing", "yahoo"], [("bing", "timeout")], 3.0, now=NOW + i)
    assert not table.usable("bing", now=NOW + 5)
    assert table.usable("yahoo", now=NOW + 5) and table.get("yahoo").n == 2


def test_all_failed_with_mixed_reasons_still_penalised(table: HealthTable) -> None:
    for i in range(2):
        table.record(
            ["bing", "yahoo"], [("bing", "timeout"), ("yahoo", "Suspended: CAPTCHA")], 3.0, now=NOW + i
        )
    assert not table.usable("bing", now=NOW + 5)  # two consecutive timeouts
    assert not table.usable("yahoo", now=NOW + 23 * 3600)


def test_single_engine_timeout_is_penalised(table: HealthTable) -> None:
    # the instance-wide rule needs at least two engines asked
    table.record(["yep"], [("yep", "timeout")], 3.0, now=NOW)
    assert table.get("yep").fail_streak == 1


def test_success_resets_and_good(table: HealthTable) -> None:
    for i in range(3):
        table.record(["bing"], [], 0.9, now=NOW + i)
    h = table.get("bing")
    assert h.n == 3 and h.success == pytest.approx(1.0) and table.good("bing", now=NOW + 5)
    table.record(["bing"], [("bing", "timeout")], 2.5, now=NOW + 10)
    assert table.get("bing").fail_streak == 1 and table.usable("bing", now=NOW + 11)


def test_good_ignores_request_latency(table: HealthTable) -> None:
    # the elapsed time is the whole request's, so a slow request must not mark an engine bad
    table.record(["duckduckgo", "bing"], [], 4.8, now=NOW)
    assert table.get("duckduckgo").latency == pytest.approx(4.8)  # still recorded for diagnostics
    assert table.good("duckduckgo", now=NOW + 1)


def test_good_requires_success_closed_and_data() -> None:
    assert not EngineHealth("x").good(NOW)  # no observations yet
    assert not EngineHealth("x", n=3, success=0.5).good(NOW)
    assert not EngineHealth("x", n=3, success=1.0, open_until=NOW + 60).good(NOW)
    assert EngineHealth("x", n=1, success=0.8, latency=9.0).good(NOW)


def test_overrides_and_probe(table: HealthTable) -> None:
    table.overrides = parse_overrides("yep=off, duckduckgo=on, junk")
    assert table.overrides == {"yep": False, "duckduckgo": True}
    assert not table.usable("yep") and table.usable("duckduckgo") and table.good("duckduckgo")
    assert not table.should_probe("yep")
    assert table.should_probe("brand-new-engine")


def test_cache_ttl() -> None:
    import time

    cache = Cache(Store(":memory:"))
    cache.set("ns", "k", {"a": 1}, ttl=60)
    assert cache.get("ns", "k") == {"a": 1}
    cache.set("ns", "old", 1, ttl=0.001)
    time.sleep(0.01)
    assert cache.get("ns", "old") is None
    assert Cache(Store(":memory:"), enabled=False).get("ns", "k") is None
