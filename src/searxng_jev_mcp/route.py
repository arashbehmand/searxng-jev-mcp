"""Engine selection: computed general baseline, caller hints, static keyword map, quota buckets.

Decision-model routing (stage B) lives in :mod:`searxng_jev_mcp.decide.backend`; this module holds
everything that works without it, which is also the fail-open path.

Precedence: explicit `engines` > explicit `categories` > (decision model ∪ static rules) > general baseline.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .health import HealthTable
from .models import ParsedQuery
from .searxng import Catalog
from .store import Cache

MAX_VERTICAL_ENGINES = 8
ROUTED_CATEGORIES = (
    "general",
    "news",
    "science",
    "it",
    "q&a",
    "packages",
    "repos",
    "dictionaries",
    "wikimedia",
)


@dataclass(slots=True)
class RoutingCatalog:
    categories: dict[str, dict[str, str]]
    engines: dict[str, dict[str, Any]]
    category_engines: dict[str, list[str]]
    general: dict[str, Any]
    quota: dict[str, dict[str, Any]]

    def is_quota(self, engine: str) -> bool:
        return bool(self.engines.get(engine, {}).get("quota"))

    def shape(self, engine: str) -> str | None:
        return self.engines.get(engine, {}).get("shape")

    def bucket_for(self, engine: str) -> str | None:
        for name, spec in self.quota.items():
            if engine in spec.get("engines", []):
                return name
        return None

    def knows(self, engine: str) -> bool:
        """True if this catalog has any policy for `engine` (described, listed for a category, or
        part of the general baseline)."""
        if engine in self.engines:
            return True
        if any(engine in names for names in self.category_engines.values()):
            return True
        return any(engine in (self.general.get(k) or []) for k in ("core", "conditional", "flagged"))


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> RoutingCatalog:
    path = path or os.environ.get("CATALOG_PATH")
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("searxng_jev_mcp").joinpath("catalog.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return RoutingCatalog(
        categories=data.get("categories") or {},
        engines=data.get("engines") or {},
        category_engines=data.get("category_engines") or {},
        general=data.get("general") or {},
        quota=data.get("quota") or {},
    )


# --------------------------------------------------------------------------------------------- baseline
_GENERAL_FALLBACK_SKIP = frozenset({"currency", "lingva", "dictzone", "mymemory translated", "wikidata"})


def general_baseline_with_status(
    catalog: Catalog,
    health: HealthTable,
    rc: RoutingCatalog,
    *,
    google_cse: bool = True,
    now: float | None = None,
) -> tuple[list[str], bool]:
    """General engines for request C, computed from /config and health (not hard-coded).

    Returns `(engines, probing)`. Open circuits deprioritise engines but never remove them: when
    every enabled core engine is circuit-open, they are all included anyway as a half-open probe and
    `probing` is True, so the baseline is never empty while the instance has general engines.
    Engines an operator override forces off are never probed.
    """

    def allowed(e: str) -> bool:
        return catalog.enabled(e) and health.overrides.get(e) is not False

    core_enabled = [e for e in rc.general.get("core", []) if allowed(e)]
    core = [e for e in core_enabled if health.usable(e, now)]
    conditional = [
        e
        for e in rc.general.get("conditional", [])
        if catalog.enabled(e) and (health.good(e, now) or health.should_probe(e, now))
    ]
    flagged = [
        e
        for e in rc.general.get("flagged", [])
        if google_cse and catalog.enabled(e) and health.usable(e, now)
    ]
    probing = False
    if core_enabled and not core:
        core, probing = core_enabled, True
    chosen = list(dict.fromkeys(core + conditional + flagged))
    if not chosen:
        # Unknown instance layout: every enabled general engine that is not a converter/translator
        # (flagged engines only while their flag is on).
        skip = _GENERAL_FALLBACK_SKIP | (set() if google_cse else set(rc.general.get("flagged") or []))
        pool = [e for e in catalog.in_category("general") if e not in skip and allowed(e)]
        chosen = [e for e in pool if health.usable(e, now)]
        if not chosen and pool:
            chosen, probing = pool, True
    return chosen, probing


def general_baseline(
    catalog: Catalog,
    health: HealthTable,
    rc: RoutingCatalog,
    *,
    google_cse: bool = True,
    now: float | None = None,
) -> list[str]:
    """General engines for request C; see :func:`general_baseline_with_status`."""
    return general_baseline_with_status(catalog, health, rc, google_cse=google_cse, now=now)[0]


# --------------------------------------------------------------------------------------------- static map
_CODE_TOKEN = re.compile(
    r"(::|\(\)|\b[a-z]+[A-Z][A-Za-z]+\b|\b[a-z]+_[a-z_]+\b|\.(py|js|ts|rs|go|java|cpp|rb|sh|yml|yaml|toml|json)\b"
    r"|\b[A-Z]{1,4}\d{3,5}\b|\b[A-Z][A-Z_]{4,}_[A-Z]+\b|\b0x[0-9a-fA-F]{8}\b)"
)
_STATIC_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "it",
        re.compile(
            r"\b(error|exception|traceback|stack ?trace|segfault|compile[rd]?|deprecat\w*|api|sdk|cli|library|"
            r"function|method|class|module|import|syntax|regex|python|javascript|typescript|node(\.js)?|rust|golang|"
            r"java|kotlin|swift|c\+\+|c#|\.net|php|ruby|bash|zsh|shell|powershell|sql|postgres(ql)?|mysql|sqlite|"
            r"redis|docker|kubernetes|k8s|nginx|git|linux|macos|windows 1[01]|npm|pip|cargo|gradle|maven|webpack|"
            r"vite|react|vue|svelte|django|flask|fastapi|pandas|numpy|pytorch|tensorflow|llvm|gcc|clang|wasm|"
            r"http|https|tls|ssl|dns|tcp|json|yaml|toml|css|html|dom)\b",
            re.I,
        ),
    ),
    (
        "q&a",
        re.compile(
            r"\b(how (do|to|can)|why (does|is|do)|not working|fails?|failed|unable to|cannot|can't)\b", re.I
        ),
    ),
    (
        "science",
        re.compile(
            r"\b(paper|papers|preprint|arxiv|doi|pubmed|study|studies|meta-analysis|systematic review|clinical|"
            r"trial|theorem|lemma|proof|et al|journal|peer[- ]reviewed|dataset|benchmark|survey of)\b",
            re.I,
        ),
    ),
    (
        "packages",
        re.compile(r"\b(pypi|package|pip install|docker image|container image|wheel|hackage|hoogle)\b", re.I),
    ),
    (
        "repos",
        re.compile(
            r"\b(github|repo|repository|source code|open[- ]source (library|tool|implementation))\b", re.I
        ),
    ),
    (
        "news",
        re.compile(
            r"\b(news|breaking|announced|announcement|election|today|yesterday|this week|headline|press release)\b",
            re.I,
        ),
    ),
    (
        "dictionaries",
        re.compile(r"\b(define|definition|meaning of|etymology|synonyms?|antonyms?|pronunciation)\b", re.I),
    ),
]
_ENGINE_GATES: dict[str, re.Pattern[str]] = {
    "mdn": re.compile(
        r"\b(html|css|javascript|js|dom|web ?api|fetch|http|browser|svg|wasm|typescript)\b", re.I
    ),
    "docker hub": re.compile(r"\b(docker|container|image|compose)\b", re.I),
    "pypi": re.compile(r"\b(python|pip|pypi|package|wheel|py)\b", re.I),
    "hoogle": re.compile(r"\b(haskell|ghc|hoogle|hackage)\b", re.I),
    "arch linux wiki": re.compile(r"\b(linux|arch|systemd|pacman|xorg|wayland|grub|kernel)\b", re.I),
    "pubmed": re.compile(
        r"\b(clinical|patient|disease|medical|drug|gene|protein|cell|cancer|trial|health)\b", re.I
    ),
    "europepmc": re.compile(
        r"\b(clinical|patient|disease|medical|drug|gene|protein|cell|cancer|trial|health)\b", re.I
    ),
    "askubuntu": re.compile(r"\b(ubuntu|debian|apt|linux|gnome|snap)\b", re.I),
    "superuser": re.compile(r"\b(windows|macos|excel|word|keyboard|laptop|bios|usb|wifi|bluetooth)\b", re.I),
}


def static_categories(pq: ParsedQuery) -> list[str]:
    """Keyword/shape rules → candidate vertical categories (recall-biased, cheap, offline)."""
    text = pq.lexical_text
    cats: list[str] = []
    if _CODE_TOKEN.search(pq.original):
        cats.append("it")
    for cat, rx in _STATIC_RULES:
        if rx.search(text) and cat not in cats:
            cats.append(cat)
    if "it" in cats and "q&a" not in cats and _CODE_TOKEN.search(pq.original):
        cats.append("q&a")
    if "q&a" in cats and "it" not in cats:
        # "how to ..." alone is not a programming question; Stack Exchange quota is shared and small
        cats.remove("q&a")
    return cats


def engine_gate_ok(engine: str, pq: ParsedQuery) -> bool:
    gate = _ENGINE_GATES.get(engine)
    return gate is None or bool(gate.search(pq.lexical_text))


def engines_for_categories(
    categories: list[str],
    pq: ParsedQuery,
    catalog: Catalog,
    health: HealthTable,
    rc: RoutingCatalog,
    *,
    gated: bool = True,
    cap: int = MAX_VERTICAL_ENGINES,
    exclude: set[str] | None = None,
    include_unlisted: bool = False,
    now: float | None = None,
) -> list[str]:
    """Pick vertical engines for categories in order, respecting enabled/health/gates and the cap.

    A category listed in `catalog.yaml` uses only its listed engines, unless `include_unlisted`:
    then enabled instance engines of that category that the routing catalog does not know at all
    follow the listed ones (for explicit caller categories, so the YAML never hides an instance's
    engines). Leave it off for static/decision routing: unknown engines have no keyword gate.
    """
    out: list[str] = []
    exclude = exclude or set()
    for cat in categories:
        listed = list(rc.category_engines.get(cat) or catalog.in_category(cat))
        if include_unlisted and rc.category_engines.get(cat):
            listed += [e for e in catalog.in_category(cat) if e not in listed and not rc.knows(e)]
        for e in listed:
            if len(out) >= cap:
                return out
            if e in out or e in exclude or not catalog.enabled(e) or not health.usable(e, now):
                continue
            if gated and not engine_gate_ok(e, pq):
                continue
            out.append(e)
    return out


def available_verticals(
    pq: ParsedQuery,
    catalog: Catalog,
    health: HealthTable,
    rc: RoutingCatalog,
    *,
    asked_engines: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Routable categories → engines the decision model may authorise. Engines that get their own
    engine-level question (`asked_engines`) are judged by it; the rest (e.g. Arch wiki, Hoogle) must
    also pass their keyword gate, or they would ride along with every query in their category."""
    out: dict[str, list[str]] = {}
    for cat in ROUTED_CATEGORIES:
        if cat == "general":
            continue
        engines = engines_for_categories([cat], pq, catalog, health, rc, gated=False)
        engines = [e for e in engines if e in asked_engines or engine_gate_ok(e, pq)]
        if engines:
            out[cat] = engines
    return out


def caller_engines(engines: list[str], catalog: Catalog) -> tuple[list[str], list[str]]:
    """Validate caller-named engines. Returns (usable, unknown_or_disabled). Disabled engines are refused
    because SearXNG would query them anyway (they are usually disabled for being blocked)."""
    ok, bad = [], []
    for e in engines:
        name = e.strip().lower()
        (ok if catalog.enabled(name) else bad).append(name)
    return ok, bad


# --------------------------------------------------------------------------------------------- quota
@dataclass(slots=True)
class QuotaBuckets:
    """Fixed-window call counters per bucket, persisted in the cache table.

    Every engine counts separately: stackoverflow, askubuntu and superuser in one SearXNG request
    are three Stack Exchange API calls against the same bucket.
    """

    cache: Cache
    rc: RoutingCatalog
    _local: dict[str, int] = field(default_factory=dict)

    def _key(self, bucket: str, now: float) -> tuple[str, float]:
        window = float(self.rc.quota[bucket].get("window_s", 86400))
        start = int(now // window)
        return f"{bucket}:{start}", window - (now % window)

    def _used(self, key: str) -> int:
        return max(int(self.cache.get("quota", key) or 0), self._local.get(key, 0))

    def _limit(self, bucket: str) -> int:
        return int(self.rc.quota[bucket].get("limit", 0))

    def allow(self, engine: str, now: float | None = None) -> bool:
        bucket = self.rc.bucket_for(engine)
        if not bucket:
            return True
        key, _ = self._key(bucket, now or time.time())
        return self._used(key) < self._limit(bucket)

    def filter(self, engines: list[str], now: float | None = None) -> tuple[list[str], list[str]]:
        """Split `engines` (one request) into (allowed, skipped), counting engines of the same bucket
        against each other: with 249 of 250 used, only the first of three Stack Exchange engines
        is allowed. Duplicates are dropped; order is kept."""
        now = now or time.time()
        allowed: list[str] = []
        skipped: list[str] = []
        pending: dict[str, int] = {}
        for e in dict.fromkeys(engines):
            bucket = self.rc.bucket_for(e)
            if not bucket:
                allowed.append(e)
                continue
            key, _ = self._key(bucket, now)
            if self._used(key) + pending.get(key, 0) < self._limit(bucket):
                pending[key] = pending.get(key, 0) + 1
                allowed.append(e)
            else:
                skipped.append(e)
        return allowed, skipped

    def spend(self, engines: list[str], now: float | None = None) -> None:
        """Count one call per quota engine actually asked (duplicates in `engines` count once)."""
        now = now or time.time()
        for e in dict.fromkeys(engines):
            bucket = self.rc.bucket_for(e)
            if not bucket:
                continue
            key, ttl = self._key(bucket, now)
            used = self._used(key) + 1
            self._local[key] = used
            self.cache.set("quota", key, used, ttl)


def google_cse_active(enabled: bool, today: dt.date | None = None) -> bool:
    return enabled and (today or dt.date.today()) < dt.date(2027, 1, 1)
