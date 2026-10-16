"""Per-engine health: EWMA success, plus a circuit breaker.

Fed from each response's `unresponsive_engines`. Reasons are matched loosely because SearXNG
translates them (the client also pins `locale=en`). Open circuits deprioritise an engine (it is
left out of automatic selection) but never remove it: explicit `engines=` still reaches it, and a
half-open probe re-admits it after the cool-down.

Latency is recorded for diagnostics only: SearXNG reports no per-engine timing, so the stored value
is the elapsed time of the whole request each engine took part in, and it plays no part in `good()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Store

ALPHA = 0.3  # EWMA weight of the newest observation
FAIL_THRESHOLD = 2  # consecutive timeouts/errors before the circuit opens
SHORT_OPEN = 15 * 60  # seconds, after FAIL_THRESHOLD consecutive timeouts/errors
HOUR_OPEN = 3600  # rate limits, access denied, other SearXNG suspensions: usually transient
LONG_OPEN = 24 * 3600  # CAPTCHA: the engine stays challenged for a long time
PROBE_INTERVAL = 30 * 60  # unknown or stale engines are probed at most this often

# Reason classes, checked in this order. A GitHub API rate limit arrives as HTTP 403 ("access
# denied"), so 403 must not mean a day-long block; only a CAPTCHA does.
_CAPTCHA_WORDS = ("captcha",)
_RATE_WORDS = ("too many requests", "429", "rate limit", "rate-limit", "ratelimit", "quota")
_DENIED_WORDS = ("access denied", "forbidden", "403", "blocked")
_TIMEOUT_WORDS = ("timeout", "timed out")

# Classes that open the circuit on the first failure, with their cool-down in seconds.
# `timeout` and `error` open only after FAIL_THRESHOLD consecutive failures (SHORT_OPEN).
COOL_DOWN: dict[str, float] = {
    "captcha": LONG_OPEN,
    "rate_limited": HOUR_OPEN,
    "blocked": HOUR_OPEN,
    "suspended": HOUR_OPEN,
}


@dataclass(slots=True)
class EngineHealth:
    engine: str
    success: float = 1.0
    latency: float = 0.0  # request-level elapsed time (diagnostic only, see module docstring)
    n: int = 0
    fail_streak: int = 0
    open_until: float = 0.0
    last_reason: str | None = None
    last_probe: float = 0.0

    def is_open(self, now: float | None = None) -> bool:
        return self.open_until > (now or time.time())

    def good(self, now: float | None = None, *, min_success: float = 0.75) -> bool:
        return self.n >= 1 and not self.is_open(now) and self.success >= min_success


def classify_reason(reason: str) -> str:
    """Map a SearXNG `unresponsive_engines` reason to one of: captcha, rate_limited, blocked,
    suspended, timeout, error."""
    r = reason.lower()
    if any(w in r for w in _CAPTCHA_WORDS):
        return "captcha"
    if any(w in r for w in _RATE_WORDS):
        return "rate_limited"
    if any(w in r for w in _DENIED_WORDS):
        return "blocked"
    if "suspended" in r:
        return "suspended"
    if any(w in r for w in _TIMEOUT_WORDS):
        return "timeout"
    return "error"


class HealthTable:
    def __init__(self, store: Store, overrides: dict[str, bool] | None = None):
        self.store = store
        self.overrides = overrides or {}  # engine -> forced on (True) / off (False)

    def get(self, engine: str) -> EngineHealth:
        rows = self.store.execute("SELECT * FROM engine_health WHERE engine=?", (engine,))
        if not rows:
            return EngineHealth(engine)
        r = rows[0]
        return EngineHealth(
            engine=r["engine"],
            success=r["success"],
            latency=r["latency"],
            n=r["n"],
            fail_streak=r["fail_streak"],
            open_until=r["open_until"],
            last_reason=r["last_reason"],
            last_probe=r["last_probe"],
        )

    def all(self) -> list[EngineHealth]:
        return [
            self.get(r["engine"])
            for r in self.store.execute("SELECT engine FROM engine_health ORDER BY engine")
        ]

    def _save(self, h: EngineHealth) -> None:
        self.store.execute(
            "INSERT OR REPLACE INTO engine_health VALUES (?,?,?,?,?,?,?,?,?)",
            (
                h.engine,
                h.success,
                h.latency,
                h.n,
                h.fail_streak,
                h.open_until,
                h.last_reason,
                h.last_probe,
                time.time(),
            ),
        )

    def record(
        self, asked: list[str], unresponsive: list[tuple[str, str]], elapsed: float, now: float | None = None
    ) -> None:
        """Update every asked engine from one response.

        When two or more engines were asked and every one of them timed out, the cause is almost
        always SearXNG's own network or load rather than the engines, so nobody is penalised: only
        `last_probe` moves (which still spaces out half-open probes).
        """
        now = now or time.time()
        engines = list(dict.fromkeys(asked))
        failed = {name: classify_reason(reason) for name, reason in unresponsive}
        if len(engines) >= 2 and all(failed.get(e) == "timeout" for e in engines):
            for engine in engines:
                h = self.get(engine)
                h.last_probe = now
                self._save(h)
            return
        for engine in engines:
            h = self.get(engine)
            h.last_probe = now
            kind = failed.get(engine)
            if kind is not None:
                h.success = (1 - ALPHA) * h.success if h.n else 0.0
                h.fail_streak += 1
                h.last_reason = kind
                cool = COOL_DOWN.get(kind)
                if cool is None and h.fail_streak >= FAIL_THRESHOLD:
                    cool = SHORT_OPEN
                if cool is not None:
                    # a weaker failure never shortens a cool-down already running (e.g. a CAPTCHA)
                    h.open_until = max(h.open_until, now + cool)
                if kind == "timeout":
                    h.latency = (1 - ALPHA) * h.latency + ALPHA * elapsed if h.n else elapsed
            else:
                h.success = (1 - ALPHA) * h.success + ALPHA * 1.0 if h.n else 1.0
                h.latency = (1 - ALPHA) * h.latency + ALPHA * elapsed if h.n else elapsed
                h.fail_streak = 0
                h.open_until = 0.0
                h.last_reason = None
            h.n += 1
            self._save(h)

    def usable(self, engine: str, now: float | None = None) -> bool:
        """False while the circuit is open (unless an override forces the engine on)."""
        if engine in self.overrides:
            return self.overrides[engine]
        return not self.get(engine).is_open(now)

    def should_probe(self, engine: str, now: float | None = None) -> bool:
        """Half-open probe for engines that are unknown, stale, or whose circuit just closed."""
        now = now or time.time()
        if self.overrides.get(engine) is False:
            return False
        h = self.get(engine)
        if h.is_open(now):
            return False
        return now - h.last_probe >= PROBE_INTERVAL

    def good(self, engine: str, now: float | None = None) -> bool:
        if engine in self.overrides:
            return self.overrides[engine]
        return self.get(engine).good(now)


def parse_overrides(spec: str | None) -> dict[str, bool]:
    """`"yep=off,duckduckgo=on"` → {"yep": False, "duckduckgo": True}."""
    out: dict[str, bool] = {}
    for part in (spec or "").split(","):
        if "=" not in part:
            continue
        name, _, val = part.partition("=")
        name, val = name.strip().lower(), val.strip().lower()
        if name and val in ("on", "off", "1", "0", "true", "false"):
            out[name] = val in ("on", "1", "true")
    return out
