"""The search pipeline: stages A-H with deadlines, speculation and fail-open degradation.

    A parse/scrub → (B route ∥ C general) → D verticals → E merge → F prior → G rerank → H assess

`fast` runs without any decision model (static routing, prior ranking). `balanced` adds decision
routing and reranking when a provider is configured; every decision stage fails open to the prior.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from . import lexical, merge, query, route
from .health import HealthTable
from .models import Doc, ParsedQuery, RawResult
from .sanitize import looks_like_injection
from .searxng import Catalog, SearxngClient, SearxngError, SearxResponse
from .store import Cache, TraceLog

Mode = Literal["balanced", "fast"]
TIME_RANGES = ("day", "week", "month", "year")

# Latency budget. Stage deadlines for hosted providers; a local provider (Ollaya on CPU) brings
# its own, longer deadlines and an extra total budget through its profile (Decider.deadlines).
TIMEOUT_LIMIT = {"balanced": 2.5, "fast": 2.0}
TOTAL_DEADLINE = {"balanced": 6.0, "fast": 4.0}
ROUTE_DEADLINE = 0.8
RERANK_DEADLINE = 2.0
ASSESS_DEADLINE = 1.5
POOL_NS = "pool:v2"  # bump when Doc/ParsedQuery change shape; old entries are then ignored
SUSPECT_DEMOTION = 5  # places an injection-flagged result moves down

# Ranking constants. Initial values were k=20, BM25F weight 0.7; tuning on the dev split chose
# k=60, weight 1.0 (dev nDCG@10 0.704 → 0.763, confirmed on test 0.733 → 0.769). See docs/evaluation.md.
RRF_K = 60
BM25_WEIGHT = 1.0
TIE_EPS = 0.02  # |Δp| below this, the prior breaks ties (about Jev's run-to-run drift)


@dataclass(slots=True)
class SearchParams:
    query: str
    k: int = 8
    mode: Mode = "balanced"
    categories: list[str] | None = None
    engines: list[str] | None = None
    site: str | None = None
    exclude_domains: list[str] | None = None
    time_range: str | None = None
    language: str | None = None
    queries: list[str] | None = None
    page: int = 1
    assess: bool = False
    private: bool = False
    raw: bool = False
    explain: bool = False

    def pool_key(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("page")
        d.pop("explain")
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:32]


@dataclass(slots=True)
class Warn:
    code: str
    effect: str


class ParamError(ValueError):
    """Invalid parameters; the message lists allowed values."""


@dataclass(slots=True)
class RouteDecision:
    """Output of stage B (decision routing)."""

    categories: list[str]
    engines: list[str]
    keep_general: bool
    navigational: float
    needs_recent: float
    probabilities: dict[str, float]
    vetoed: list[str] = field(default_factory=list)  # engines the model explicitly said no to


@dataclass(slots=True)
class AssessResult:
    verdict: str
    answered: float
    ambiguous: float
    has_primary: float
    hint: str | None = None


class Decider(Protocol):
    """What the pipeline needs from a decision backend (implemented in decide/)."""

    @property
    def name(self) -> str: ...

    @property
    def candidates_k(self) -> int: ...

    @property
    def deadlines(self) -> tuple[float, float, float, float]:
        """(route, rerank, assess, extra total budget) in seconds."""
        ...

    @property
    def keep_threshold(self) -> float: ...

    @property
    def navigational_threshold(self) -> float: ...

    @property
    def thresholds(self) -> Any: ...

    async def route(
        self, pq: ParsedQuery, available: dict[str, list[str]], deadline: float
    ) -> RouteDecision: ...

    async def rerank(
        self, pq: ParsedQuery, docs: list[Doc], *, navigational: bool, deadline: float
    ) -> dict[str, Any]: ...

    async def assess(
        self, pq: ParsedQuery, docs: list[Doc], route: RouteDecision | None, deadline: float
    ) -> AssessResult: ...


@dataclass(slots=True)
class SearchOutcome:
    params: SearchParams
    pq: ParsedQuery
    docs: list[Doc]
    pool_size: int
    answers: list[str] = field(default_factory=list)
    infoboxes: list[dict[str, Any]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    warnings: list[Warn] = field(default_factory=list)
    ranker: Literal["jev", "prior"] = "prior"
    router: Literal["jev", "static", "caller"] = "static"
    degraded: bool = False
    partial: bool = False
    weak: bool = False
    has_more: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    engines_asked: list[str] = field(default_factory=list)
    engines_failed: list[tuple[str, str]] = field(default_factory=list)
    routing: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    decision_provider: str | None = None


# --------------------------------------------------------------------------------------------- helpers
def validate(params: SearchParams, catalog: Catalog) -> None:
    if not params.query or not params.query.strip():
        raise ParamError("`query` must be a non-empty string.")
    if not 1 <= params.k <= 20:
        raise ParamError("`k` must be between 1 and 20.")
    if params.mode not in ("balanced", "fast"):
        raise ParamError("`mode` must be one of: balanced, fast.")
    if params.time_range is not None and params.time_range not in TIME_RANGES:
        raise ParamError(f"`time_range` must be one of: {', '.join(TIME_RANGES)}.")
    if params.page < 1 or params.page > 10:
        raise ParamError("`page` must be between 1 and 10.")
    if params.queries and len(params.queries) > 3:
        raise ParamError("`queries` accepts at most 3 extra variants.")
    if params.engines is not None and not any(e.strip() for e in params.engines):
        raise ParamError("`engines` must name at least one engine (or be omitted).")
    if params.categories:
        known = set(catalog.categories())
        bad = [c for c in params.categories if c not in known]
        if bad:
            raise ParamError(
                f"Unknown categories {bad}. Allowed on this instance: {', '.join(sorted(known & set(route.ROUTED_CATEGORIES)) or ', '.join(sorted(known)))}."
            )


def cache_ttl(resp: SearxResponse, time_range: str | None, labels_news: bool) -> float:
    if resp.unresponsive or not resp.results:
        return 60.0
    if time_range or labels_news:
        return 600.0
    return 6 * 3600.0


def age_string(published: dt.datetime | None, now: dt.datetime | None = None) -> str | None:
    """Human age computed in code ("8 months"); raw dates are never sent to the decision model."""
    if not published:
        return None
    now = now or dt.datetime.now(dt.UTC)
    if published.tzinfo is None:
        published = published.replace(tzinfo=dt.UTC)
    days = (now - published).days
    if days < 0:
        return None
    if days < 1:
        return "today"
    if days < 14:
        return f"{days} day{'s' if days != 1 else ''}"
    if days < 60:
        return f"{days // 7} weeks"
    if days < 730:
        return f"{days // 30} months"
    return f"{days // 365} years"


def group_unresponsive(failed: list[tuple[str, str]]) -> list[Warn]:
    """One warning per reason class: [yep, duckduckgo: timeout]."""
    from .health import classify_reason

    groups: dict[str, list[str]] = {}
    for name, reason in failed:
        groups.setdefault(classify_reason(reason), []).append(name)
    out = []
    for kind, names in sorted(groups.items()):
        out.append(Warn(f"engines_{kind}", f"no results from {', '.join(sorted(set(names)))} ({kind})"))
    return out


def doc_to_dict(d: Doc) -> dict[str, Any]:
    out = dataclasses.asdict(d)
    out["engines"] = sorted(d.engines)
    out["published"] = d.published.isoformat() if d.published else None
    return out


def doc_from_dict(x: dict[str, Any]) -> Doc:
    """Raises TypeError/KeyError/ValueError on a stale cache shape; callers treat that as a miss."""
    x = dict(x)
    x["engines"] = set(x.get("engines") or [])
    x["published"] = dt.datetime.fromisoformat(x["published"]) if x.get("published") else None
    return Doc(**x)


def response_to_dict(r: SearxResponse) -> dict[str, Any]:
    out = dataclasses.asdict(r)
    for row in out["results"]:
        row["published"] = row["published"].isoformat() if row["published"] else None
    return out


def response_from_dict(x: dict[str, Any]) -> SearxResponse:
    rows = []
    for row in x["results"]:
        row = dict(row)
        row["published"] = dt.datetime.fromisoformat(row["published"]) if row.get("published") else None
        rows.append(RawResult(**row))
    x = {**x, "results": rows, "unresponsive": [tuple(p) for p in x["unresponsive"]], "from_cache": True}
    return SearxResponse(**x)


def vertical_q(pq: ParsedQuery) -> str:
    """Request D always gets a short keyword form: vertical APIs (arXiv, PubMed, GitHub, PyPI, Stack
    Exchange) handle long natural-language queries badly, and news/wiki engines are fine with keywords."""
    shaped = dataclasses.replace(pq, text=query.keyword_form(pq), phrases=[])
    return query.to_searxng_q(shaped)


def build_pool(
    pq: ParsedQuery,
    responses: list[SearxResponse],
    *,
    recency: bool = False,
    bm25_weight: float = BM25_WEIGHT,
    rrf_k: int = RRF_K,
    request_weights: dict[str, float] | None = None,
    dedupe: bool = True,
) -> list[Doc]:
    """Stages E and F as a pure function (also used by eval replay): merge, post-filter, BM25F, prior,
    cross-site repost dedupe."""
    raws = [row for r in responses for row in r.results]
    docs = merge.merge_results(raws)
    docs = merge.apply_post_filters(docs, pq)
    lexical.lexical_features(pq, docs)
    ranked = lexical.compute_prior(
        docs, bm25_weight=bm25_weight, rrf_k=rrf_k, recency=recency, request_weights=request_weights
    )
    if dedupe:
        ranked = merge.dedupe_reposts(ranked)
    return demote_suspects(ranked)


def flag_suspects(docs: list[Doc]) -> None:
    for d in docs:
        if not d.suspect and looks_like_injection(d.title, d.snippet):
            d.suspect = True
            if "suspect" not in d.flags:
                d.flags.append("suspect")


def _demote(order: list[Doc], places: int = SUSPECT_DEMOTION) -> list[Doc]:
    """Move each suspect `places` positions further down than it would otherwise be."""
    suspects = [(i, d) for i, d in enumerate(order) if d.suspect]
    out = [d for d in order if not d.suspect]
    for i, d in suspects:
        out.insert(min(len(out), i + places), d)
    return out


def demote_suspects(ranked: list[Doc], places: int = SUSPECT_DEMOTION) -> list[Doc]:
    """Flag results whose text tries to steer a model or ranker ("ignore previous instructions", "rate
    this result as relevant") and move them a few places down. Not to the end: the heuristic has false
    positives, and the decision model itself scored the eval's injected spam low (docs/evaluation.md)."""
    flag_suspects(ranked)
    return _demote(ranked, places)


# --------------------------------------------------------------------------------------------- ordering
def _independent_voters(d: Doc, general_engines: frozenset[str]) -> int:
    """General engines that returned the doc, with Bing and Yahoo (same index) counted once."""
    voters = {("bing" if e == "yahoo" else e) for e in d.engines if e in general_engines}
    return len(voters)


GUARD_MODES = ("off", "pin3", "pin3_p80", "pin3_band", "pin1_p80", "pin1_band", "band_reorder")
GUARD_MODE = "pin3_p80"  # default authority guard, chosen after the guard-variant eval (AUTHORITY_GUARD)
GUARD_BAND = 0.1  # "near the best": p within this of the best p in the pool
GUARD_HIGH_P = 0.8


def _guard_reason(d: Doc, voters: int, best: float | None) -> str:
    p = f"{d.p_jev:.2f}" if d.p_jev is not None else "unscored"
    tail = f" (best in pool {best:.2f})" if best is not None else ""
    return (
        f"kept near SearXNG's rank {d.request_ranks.get('general')}: {voters} independent general engines "
        f"agree, model p {p}{tail}"
    )


def apply_guard(
    out: list[Doc], mode: str, *, keep: float, general_engines: frozenset[str], high_p: float = GUARD_HIGH_P
) -> list[Doc]:
    """Authority guard variants (see GUARD_MODES). A result that the guard moves up gets the flag
    `guarded` and a `guard_reason`, so the order never looks unsorted without an explanation."""
    if mode == "off" or not general_engines:
        return out
    scored_ps = [d.p_jev for d in out if d.p_jev is not None and not d.suspect]
    best = max(scored_ps) if scored_ps else None
    max_rank = 1 if mode.startswith("pin1") else 3

    def agreed(d: Doc) -> int:
        if d.suspect or d.request_ranks.get("general", 99) > max_rank:
            return 0
        v = _independent_voters(d, general_engines)
        return v if v >= 2 else 0

    def condition(d: Doc) -> bool:
        if mode == "pin3":
            return d.p_jev is None or d.p_jev >= keep
        if d.p_jev is None:
            return False
        if mode.endswith("_p80"):
            return d.p_jev >= high_p  # "high" on the active model's scale (0.8 for Jev)
        return best is not None and best - d.p_jev <= GUARD_BAND  # *_band and band_reorder

    out = list(out)
    if mode == "band_reorder":
        band_pos = [
            i
            for i, d in enumerate(out)
            if d.p_jev is not None and best is not None and best - d.p_jev <= GUARD_BAND
        ]
        members = [out[i] for i in band_pos]
        lead = sorted((d for d in members if agreed(d)), key=lambda d: d.request_ranks.get("general", 99))
        rest = [d for d in members if not agreed(d)]
        orig = {d.id: i for i, d in enumerate(out)}
        for pos, d in zip(band_pos, lead + rest, strict=True):
            if orig[d.id] > pos:
                d.guard_reason = _guard_reason(d, agreed(d), best)
                if "guarded" not in d.flags:
                    d.flags.append("guarded")
            out[pos] = d
        return out
    guarded = sorted(
        (d for d in out if agreed(d) and condition(d)), key=lambda d: d.request_ranks.get("general", 99)
    )
    for d in guarded:
        target = d.request_ranks["general"] - 1
        if out.index(d) > target:
            out.remove(d)
            out.insert(target, d)
            d.guard_reason = _guard_reason(d, agreed(d), best)
            if "guarded" not in d.flags:
                d.flags.append("guarded")
    return out


def order_by_decision(
    candidates: list[Doc],
    pq: ParsedQuery | None = None,
    *,
    keep: float = 0.2,
    navigational: bool = False,
    guard: str | None = None,
    general_engines: frozenset[str] = frozenset(),
    high_p: float = GUARD_HIGH_P,
) -> list[Doc]:
    """Final order after the decision rerank.

    1. sort by p; within |Δp| < TIE_EPS of a group's top result the prior decides (Jev's run-to-run drift);
    2. results whose batch missed the deadline go back to their prior slot;
    3. results with p < keep sort after the rest (never dropped);
    4. authority guard (`guard`, default GUARD_MODE): see apply_guard and docs/evaluation.md;
    5. injection-flagged results move SUSPECT_DEMOTION places down.

    `navigational` only switches the rerank criterion and waives the diversity cap (a navigational
    "pin the entity's own site" rule was evaluated and lowered known-item MRR).
    """
    scored = sorted(
        (d for d in candidates if d.p_jev is not None and not d.suspect),
        key=lambda d: (-(d.p_jev or 0.0), d.prior_rank or 10**6),
    )
    ordered: list[Doc] = []
    i = 0
    while i < len(scored):
        head = scored[i].p_jev or 0.0
        j = i
        while j < len(scored) and head - (scored[j].p_jev or 0.0) < TIE_EPS:
            j += 1
        ordered += sorted(scored[i:j], key=lambda d: d.prior_rank or 10**6)
        i = j
    for d in sorted(
        (d for d in candidates if d.p_jev is None and not d.suspect), key=lambda d: d.prior_rank or 10**6
    ):
        ordered.insert(min(len(ordered), max(0, (d.prior_rank or len(ordered)) - 1)), d)
    low = [d for d in ordered if d.p_jev is not None and d.p_jev < keep]
    low_ids = {d.id for d in low}
    out = [d for d in ordered if d.id not in low_ids] + low
    out = apply_guard(out, guard or GUARD_MODE, keep=keep, general_engines=general_engines, high_p=high_p)
    # suspects: their prior position, then demoted
    for d in sorted((d for d in candidates if d.suspect), key=lambda d: d.prior_rank or 10**6):
        out.insert(min(len(out), (d.prior_rank or len(out)) - 1 + SUSPECT_DEMOTION), d)
    return out


def _content_words(text: str) -> set[str]:
    return {
        w for w in lexical.normalize(text).split() if w.isalpha() and len(w) >= 2 and w not in query.STOPWORDS
    }


def rel_band(p: float, high: float = 0.8, medium: float = 0.4) -> Literal["high", "medium", "low"]:
    return "high" if p >= high else "medium" if p >= medium else "low"


def relevant_suggestions(pq: ParsedQuery, suggestions: list[str], docs: list[Doc] | None = None) -> list[str]:
    """Keep a SearXNG suggestion only if
    (1) it shares an alphabetic, non-stopword word with the query (years and numbers do not count), and
    (2) when result documents are given, every word it adds to the query appears in the top results'
        titles or snippets: a refinement should stay inside the topic the results are about.
    Rule 2 is what removes "fastapi lifespan events 2026 printable / calendar / summer": the engines read
    "2026" as a date search, and none of those words occur in pages about FastAPI lifespan events."""
    q = _content_words(pq.lexical_text or pq.text)
    grounded = set().union(*(_content_words(f"{d.title} {d.snippet}") for d in docs[:10])) if docs else None
    out = []
    for s in suggestions:
        words = _content_words(s)
        if not words & q:
            continue
        if grounded is not None and not (words - q) <= grounded:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(
        self,
        searx: SearxngClient,
        health: HealthTable,
        cache: Cache,
        *,
        decider: Decider | None = None,
        private_decider: Decider | None = None,
        trace: TraceLog | None = None,
        google_cse: bool = True,
        decider_error: str | None = None,
        decision_routing: bool = False,
        guard_mode: str = GUARD_MODE,
    ):
        self.searx = searx
        self.health = health
        self.cache = cache
        self.decider = decider
        self.private_decider = private_decider
        self.trace = trace
        self.google_cse = google_cse
        self.decider_error = decider_error  # why no decider is configured, for warnings
        self.decision_routing = decision_routing
        if guard_mode not in GUARD_MODES:
            raise ValueError(f"unknown authority guard {guard_mode!r}; one of {', '.join(GUARD_MODES)}")
        self.guard_mode = guard_mode
        self.rc = route.load_catalog()
        self.quota = route.QuotaBuckets(cache, self.rc)
        g = self.rc.general
        self.general_engines = frozenset(g.get("core", []) + g.get("conditional", []) + g.get("flagged", []))
        self._searches = 0

    # ------------------------------------------------------------------ SearXNG with cache/health
    async def _fetch(
        self,
        pq: ParsedQuery,
        q: str,
        engines: list[str],
        label: str,
        *,
        time_range: str | None,
        timeout_limit: float,
        page: int = 1,
    ) -> SearxResponse:
        key = json.dumps([q, sorted(engines), pq.language, time_range, page], ensure_ascii=False)
        cached = self.cache.get("searx", key)
        if cached is not None:
            try:
                resp = response_from_dict(cached)
            except (TypeError, KeyError, ValueError):
                resp = None
            if resp is not None:
                resp.label = label
                for r in resp.results:
                    r.request = label
                return resp
        resp = await self.searx.search(
            q, engines=engines, label=label, language=pq.language, time_range=time_range,
            pageno=page, timeout_limit=timeout_limit,
        )  # fmt: skip
        self.health.record(engines, resp.unresponsive, resp.elapsed)
        self.quota.spend(engines)
        news = label == "vertical" and any(e in ("bing news", "wikinews", "qwant news") for e in engines)
        self.cache.set("searx", key, response_to_dict(resp), cache_ttl(resp, time_range, news))
        return resp

    def _choose_decider(self, params: SearchParams, warnings: list[Warn]) -> Decider | None:
        if params.mode == "fast":
            return None
        if params.private:
            if self.private_decider is None:
                warnings.append(
                    Warn(
                        "private_no_local_decider",
                        "private=true: no local decision model; prior ranking only",
                    )
                )
            return self.private_decider
        if self.decider is None:
            reason = self.decider_error or "no decision provider configured"
            warnings.append(Warn("decisions_unavailable", f"{reason}; using prior ranking"))
        return self.decider

    def _time_range_filter(
        self, engines: list[str], catalog: Catalog, time_range: str | None
    ) -> tuple[list[str], list[str]]:
        """SearXNG silently skips engines without time-range support when `time_range` is set."""
        if not time_range:
            return engines, []
        ok = [e for e in engines if catalog.engines.get(e) and catalog.engines[e].time_range_support]
        return ok, [e for e in engines if e not in ok]

    # ------------------------------------------------------------------ main entry
    async def search(self, params: SearchParams) -> SearchOutcome:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        warnings: list[Warn] = []
        catalog = await self.searx.catalog()
        validate(params, catalog)
        self._housekeeping()

        # ---- paging from the cached ranked pool
        if params.page > 1:
            pooled = self.cache.get(POOL_NS, params.pool_key())
            if pooled is not None:
                try:
                    return self._outcome_from_pool(params, pooled, t0)
                except (TypeError, KeyError, ValueError):
                    pass  # stale shape: search again

        # ---- A: parse, scrub
        pq = query.parse_query(
            params.query, raw=params.raw, language=params.language, time_range=params.time_range
        )
        if params.site:
            pq.sites = [*pq.sites, params.site.lower().removeprefix("www.")]
        for d in params.exclude_domains or []:
            pq.exclude_sites.append(d.lower().removeprefix("www."))
        if pq.scrubbed:
            warnings.append(Warn("query_scrubbed", f"removed {', '.join(pq.scrubbed)} before searching"))
        if pq.truncated:
            warnings.append(Warn("query_truncated", "query cut to ~300 characters"))
        timings["parse"] = time.perf_counter() - t0

        decider = self._choose_decider(params, warnings)
        route_dl, rerank_dl, assess_dl, extra = (
            decider.deadlines
            if decider is not None
            else (ROUTE_DEADLINE, RERANK_DEADLINE, ASSESS_DEADLINE, 0.0)
        )
        keep = decider.keep_threshold if decider is not None else 0.2
        nav_threshold = decider.navigational_threshold if decider is not None else 0.6
        mode = params.mode
        timeout_limit = TIMEOUT_LIMIT[mode]
        deadline = t0 + TOTAL_DEADLINE[mode] + extra
        q_general = query.to_searxng_q(pq)

        # ---- engine selection
        router: Literal["jev", "static", "caller"] = "static"
        general: list[str] = []
        static_vertical: list[str] = []
        if params.engines:
            router = "caller"
            ok, bad = route.caller_engines(params.engines, catalog)
            if bad:
                warnings.append(Warn("engines_refused", f"not enabled on this SearXNG: {', '.join(bad)}"))
            if not ok:
                raise ParamError(
                    f"None of the requested engines are enabled. Enabled examples: "
                    f"{', '.join(sorted(n for n, e in catalog.engines.items() if e.enabled)[:40])}."
                )
            general = ok
        elif params.categories:
            router = "caller"
            if "general" in params.categories:
                general = self._baseline(catalog, warnings)
            other = [c for c in params.categories if c != "general"]
            static_vertical = route.engines_for_categories(
                other, pq, catalog, self.health, self.rc, gated=False, include_unlisted=True
            )
            static_vertical = self._quota_filter(static_vertical, warnings)
            if other and not static_vertical:
                warnings.append(
                    Warn("categories_unavailable",
                         f"no usable engines for {', '.join(other)} (disabled, circuit-open or out of quota); "
                         "searched the general web instead")
                )  # fmt: skip
                general = general or self._baseline(catalog, warnings)
        else:
            general = self._baseline(catalog, warnings)
            static_cats = route.static_categories(pq)
            static_vertical = route.engines_for_categories(static_cats, pq, catalog, self.health, self.rc)

        if not params.categories:
            static_vertical = self._quota_filter(static_vertical, warnings)
        general, skipped_g = self._time_range_filter(general, catalog, pq.time_range)
        static_vertical, skipped_v = self._time_range_filter(static_vertical, catalog, pq.time_range)
        skipped_tr = skipped_g + skipped_v
        if pq.time_range and not general and not static_vertical:
            raise ParamError(
                f"time_range={pq.time_range}: none of the selected engines supports a time range. Engines "
                "that do: "
                + ", ".join(
                    sorted(n for n, e in catalog.engines.items() if e.enabled and e.time_range_support)
                )
            )

        # ---- C at t=0, B (decision routing) in parallel, then D, then variants (D gets the semaphore first)
        responses: list[SearxResponse] = []
        errors: list[SearxngError] = []
        tasks: dict[str, asyncio.Task[SearxResponse]] = {}
        if general:
            tasks["general"] = asyncio.create_task(
                self._fetch(
                    pq, q_general, general, "general", time_range=pq.time_range, timeout_limit=timeout_limit
                )
            )

        decision_route: RouteDecision | None = None
        vertical = list(static_vertical)
        use_routing = decider is not None and router != "caller" and self.decision_routing
        if use_routing and decider is not None:
            t_route = time.perf_counter()
            available = self._available_verticals(catalog, pq)
            try:
                decision_route = await asyncio.wait_for(
                    decider.route(pq, available, route_dl), timeout=route_dl
                )
                router = "jev"
                # the decision model decides; the static map is only the fail-open fallback
                vertical = [e for e in decision_route.engines if self.health.usable(e)]
                vertical = self._quota_filter(vertical, warnings)
                vertical, skipped_jev = self._time_range_filter(vertical, catalog, pq.time_range)
                skipped_tr += [e for e in skipped_jev if e not in skipped_tr]
                vertical = vertical[: route.MAX_VERTICAL_ENGINES]
            except Exception as e:  # timeout, provider error, circuit open: fail open to static
                warnings.append(
                    Warn(
                        "routing_fallback",
                        f"{self._who(params)}routing unavailable ({_short(e)}); static routing used",
                    )
                )
            timings["route"] = time.perf_counter() - t_route

        vertical = [e for e in vertical if e not in general]
        if vertical:
            tasks["vertical"] = asyncio.create_task(
                self._fetch(pq, vertical_q(pq), vertical, "vertical", time_range=pq.time_range,
                            timeout_limit=timeout_limit)
            )  # fmt: skip
        for i, variant in enumerate((params.queries or [])[:3], start=1):
            vpq = query.parse_query(
                variant, raw=params.raw, language=params.language, time_range=params.time_range
            )
            vpq.sites, vpq.exclude_sites = pq.sites, pq.exclude_sites
            tasks[f"variant{i}"] = asyncio.create_task(
                self._fetch(vpq, query.to_searxng_q(vpq), general or vertical, f"variant{i}",
                            time_range=pq.time_range, timeout_limit=timeout_limit)
            )  # fmt: skip
        if skipped_tr:
            warnings.append(
                Warn("time_range_skipped",
                     f"time_range={pq.time_range}: skipped {', '.join(sorted(set(skipped_tr)))} (no time-range support)")
            )  # fmt: skip
        if not tasks:
            raise SearxngError(
                "no_engines",
                "No usable SearXNG engines for this search (all disabled, circuit-open or out of quota). "
                "Try again later or name engines explicitly with `engines`.",
            )

        t_fetch = time.perf_counter()
        remaining = max(0.5, deadline - time.perf_counter() - (rerank_dl if decider else 0.0))
        done, pending = await asyncio.wait(tasks.values(), timeout=remaining)
        partial = bool(pending)
        for task in pending:
            task.cancel()
        for label, task in tasks.items():
            if task not in done:
                warnings.append(Warn("request_deadline", f"{label} request missed the deadline"))
                continue
            exc = task.exception()
            if exc is None:
                responses.append(task.result())
            elif isinstance(exc, SearxngError):
                errors.append(exc)
            else:
                errors.append(SearxngError("http", f"{type(exc).__name__}"))
        timings["searxng"] = time.perf_counter() - t_fetch

        if not responses:
            hard = [e for e in errors if e.kind in ("unreachable", "json_disabled", "rate_limited")]
            if hard:
                raise hard[0]
            if errors:
                raise errors[0]
            raise SearxngError(
                "timeout",
                f"SearXNG returned nothing within {remaining:.1f}s (a failure, not an empty result)",
            )

        # speculative general results are dropped only when the decision model is confident they don't help
        has_vertical = any(r.label == "vertical" and r.results for r in responses)
        if decision_route is not None and not decision_route.keep_general and has_vertical:
            responses = [r for r in responses if r.label != "general"]
            warnings.append(
                Warn("general_discarded", "decision routing judged general web results unhelpful")
            )

        asked = sorted({e for r in responses for e in r.engines_asked})
        failed = sorted({(n, reason) for r in responses for n, reason in r.unresponsive})
        warnings += group_unresponsive(list(failed))
        if any(r.redirected for r in responses):
            warnings.append(
                Warn("searxng_redirect", "SearXNG redirected (a !bang in a raw query); no results from it")
            )
        failed_names = {n for n, _ in failed}
        all_failed = bool(asked) and failed_names >= set(asked)
        if all_failed:
            warnings.append(
                Warn("all_engines_failed", "every engine failed; this is a failure, not an empty result")
            )

        # ---- E + F: merge, filter, BM25F, prior
        recency = pq.recency_cue or bool(decision_route and decision_route.needs_recent >= 0.5)
        ranked = build_pool(pq, responses, recency=recency)

        # ---- G: rerank
        ranker: Literal["jev", "prior"] = "prior"
        usage: dict[str, Any] = {}
        provider_used = getattr(decider, "name", None)
        navigational = bool(decision_route and decision_route.navigational >= nav_threshold)
        final = ranked
        decision_failed = False
        if decider is not None and ranked:
            t_rr = time.perf_counter()
            candidates = lexical.select_candidates(ranked, decider.candidates_k)
            cand_ids = {d.id for d in candidates}
            rest = [d for d in ranked if d.id not in cand_ids]
            budget = max(0.3, min(rerank_dl, deadline - time.perf_counter()))
            try:
                info = await decider.rerank(pq, candidates, navigational=navigational, deadline=budget)
                usage = info.get("usage", {})
                provider_used = info.get("provider") or provider_used
                if info.get("scored", 0) > 0:
                    ranker = "jev"
                    thr = decider.thresholds
                    final = order_by_decision(
                        candidates, pq, keep=keep, navigational=navigational, guard=self.guard_mode,
                        general_engines=self.general_engines, high_p=thr.guard_high,
                    ) + rest  # fmt: skip
                    for d in candidates:
                        if d.p_jev is not None and not d.suspect:
                            d.rel = rel_band(d.p_jev, thr.rel_high, thr.rel_medium)
                    for d in candidates:
                        if d.p_jev is None and not d.suspect:
                            d.flags.append("unscored")
                        elif d.p_jev is not None and d.p_jev < keep:
                            d.flags.append("low")
                    if info.get("missed", 0):
                        warnings.append(
                            Warn(
                                "rerank_partial",
                                f"{info['missed']} results kept their prior rank (late batch)",
                            )
                        )
                else:
                    decision_failed = True
                    warnings.append(
                        Warn("rerank_unavailable",
                             f"{self._who(params)}rerank failed ({info.get('error') or 'no scores'}); prior ranking used")
                    )  # fmt: skip
            except Exception as e:
                decision_failed = True
                warnings.append(
                    Warn(
                        "rerank_unavailable",
                        f"{self._who(params)}rerank failed ({_short(e)}); prior ranking used",
                    )
                )
            timings["rerank"] = time.perf_counter() - t_rr

        # ---- diversity over the whole pool (so later pages are capped too) + weak flag
        waive = bool(pq.sites) or navigational
        final = merge.diversify(final, k=len(final), waive=waive)
        for i, d in enumerate(final, start=1):
            d.final_rank = i
        weak = False
        if ranker == "jev":
            passing = [d for d in final[: params.k] if d.p_jev is not None and d.p_jev >= keep]
            weak = len(passing) < 3
            if weak:
                for d in final[: params.k]:
                    if d.p_jev is None or d.p_jev < keep:
                        d.flags.append("backfill")

        # ---- H: assess (opt-in)
        coverage = self._coverage(final[: params.k], asked, failed, weak)
        if params.assess:
            if decider is None:
                warnings.append(
                    Warn("assess_unavailable", "assess needs a decision model; deterministic coverage only")
                )
            elif final:
                t_as = time.perf_counter()
                try:
                    a = await asyncio.wait_for(
                        decider.assess(pq, final[:8], decision_route, assess_dl), timeout=assess_dl
                    )
                    coverage.update(
                        verdict=a.verdict, answered=round(a.answered, 2), ambiguous=round(a.ambiguous, 2),
                        has_primary=round(a.has_primary, 2),
                    )  # fmt: skip
                    hint = a.hint or (
                        self._broaden_hint(decision_route, responses) if a.verdict == "broaden" else None
                    )
                    if hint:
                        coverage["hint"] = hint
                except Exception as e:
                    warnings.append(Warn("assess_unavailable", f"{_short(e)}"))
                timings["assess"] = time.perf_counter() - t_as

        answers = list(dict.fromkeys(a for r in responses for a in r.answers))
        infoboxes = [ib for r in responses for ib in r.infoboxes][:2]
        suggestions = relevant_suggestions(
            pq, list(dict.fromkeys(s for r in responses for s in r.suggestions)), final
        )[:5]
        corrections = list(dict.fromkeys(c for r in responses for c in r.corrections))[:3]

        routing_failed = any(w.code == "routing_fallback" for w in warnings)
        no_decider_in_balanced = any(
            w.code in ("decisions_unavailable", "private_no_local_decider") for w in warnings
        )
        many_failed = bool(asked) and len(failed_names & set(asked)) * 2 >= len(asked)
        degraded = (
            all_failed
            or partial
            or decision_failed
            or routing_failed
            or no_decider_in_balanced
            or many_failed
        )
        timings["total"] = time.perf_counter() - t0
        page_docs = final[(params.page - 1) * params.k : params.page * params.k]
        outcome = SearchOutcome(
            params=params,
            pq=pq,
            docs=page_docs,
            pool_size=len(final),
            answers=answers[:3],
            infoboxes=infoboxes,
            suggestions=suggestions,
            corrections=corrections,
            warnings=warnings,
            ranker=ranker,
            router=router,
            degraded=degraded,
            partial=partial,
            weak=weak,
            has_more=len(final) > params.page * params.k,
            coverage=coverage,
            timings={k: round(v, 3) for k, v in timings.items()},
            engines_asked=asked,
            engines_failed=list(failed),
            routing=(dataclasses.asdict(decision_route) if decision_route else None),
            usage=usage,
            decision_provider=provider_used,
        )
        if final:
            self.cache.set(
                POOL_NS,
                params.pool_key(),
                {"docs": [doc_to_dict(d) for d in final], "meta": self._meta(outcome)},
                600.0,
            )
        self._trace(outcome)
        return outcome

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _who(params: SearchParams) -> str:
        return "local model (private=true): " if params.private else "decision "

    def _baseline(self, catalog: Catalog, warnings: list[Warn]) -> list[str]:
        engines, probing = route.general_baseline_with_status(
            catalog, self.health, self.rc, google_cse=self.google_cse
        )
        if probing:
            warnings.append(
                Warn(
                    "engines_probe", "every general engine had recently failed; asking them again as a probe"
                )
            )
        return engines

    def _quota_filter(self, engines: list[str], warnings: list[Warn]) -> list[str]:
        allowed, skipped = self.quota.filter(engines)
        if skipped:
            warnings.append(Warn("quota_skipped", f"skipped {', '.join(skipped)} (local quota reached)"))
        return allowed

    def _available_verticals(self, catalog: Catalog, pq: ParsedQuery) -> dict[str, list[str]]:
        """Routable categories → enabled, healthy engines (what the decision model may authorise)."""
        from .decide.questions import ENGINE_QUESTIONS

        return route.available_verticals(
            pq, catalog, self.health, self.rc, asked_engines=frozenset(ENGINE_QUESTIONS)
        )

    def _housekeeping(self) -> None:
        self._searches += 1
        if self._searches % 50 == 1:
            self.cache.purge_expired()

    @staticmethod
    def _coverage(
        top: list[Doc], asked: list[str], failed: list[tuple[str, str]], weak: bool
    ) -> dict[str, Any]:
        ps = [d.p_jev for d in top if d.p_jev is not None]
        failed_names = {n for n, _ in failed}
        return {
            "verdict": None,
            "top_p": round(max(ps), 2) if ps else None,
            "n_high": sum(1 for p in ps if p >= 0.8) if ps else None,
            "domains": len({d.site_key for d in top}),
            "engines_ok": len([e for e in asked if e not in failed_names]),
            "engines_asked": len(asked),
            "weak": weak,
        }

    @staticmethod
    def _broaden_hint(decision_route: RouteDecision | None, responses: list[SearxResponse]) -> str | None:
        if decision_route:
            unrouted = sorted(
                ((p, c.removeprefix("cat:")) for c, p in decision_route.probabilities.items()
                 if c.startswith("cat:") and c.removeprefix("cat:") not in decision_route.categories),
                reverse=True,
            )  # fmt: skip
            if unrouted and unrouted[0][0] >= 0.1:
                return f"try categories=['{unrouted[0][1]}']"
        for r in responses:
            if r.corrections:
                return f"try the correction: {r.corrections[0]}"
            if r.suggestions:
                return f"try: {r.suggestions[0]}"
        return None

    def _meta(self, o: SearchOutcome) -> dict[str, Any]:
        pq = dataclasses.asdict(o.pq)
        pq["original"] = ""  # the raw, unscrubbed query is never written to disk
        return {
            "answers": o.answers, "infoboxes": o.infoboxes, "suggestions": o.suggestions,
            "corrections": o.corrections, "warnings": [dataclasses.asdict(w) for w in o.warnings],
            "ranker": o.ranker, "router": o.router, "degraded": o.degraded, "weak": o.weak,
            "coverage": o.coverage, "engines_asked": o.engines_asked,
            "engines_failed": [list(p) for p in o.engines_failed], "pq": pq,
            "provider": o.decision_provider,
        }  # fmt: skip

    def _outcome_from_pool(self, params: SearchParams, pooled: dict[str, Any], t0: float) -> SearchOutcome:
        docs = [doc_from_dict(x) for x in pooled["docs"]]
        m = pooled["meta"]
        pq = ParsedQuery(**m["pq"])
        page_docs = docs[(params.page - 1) * params.k : params.page * params.k]
        return SearchOutcome(
            params=params, pq=pq, docs=page_docs, pool_size=len(docs), answers=[], infoboxes=[],
            suggestions=m["suggestions"], corrections=m["corrections"],
            warnings=[Warn(**w) for w in m["warnings"]], ranker=m["ranker"], router=m["router"],
            degraded=m["degraded"], weak=m["weak"], has_more=len(docs) > params.page * params.k,
            coverage=m["coverage"], timings={"total": round(time.perf_counter() - t0, 3), "from_pool": 1.0},
            engines_asked=m["engines_asked"], engines_failed=[tuple(p) for p in m["engines_failed"]],
            decision_provider=m.get("provider"),
        )  # fmt: skip

    def _trace(self, o: SearchOutcome) -> None:
        if not self.trace:
            return
        self.trace.write(
            {
                "q": query.to_searxng_q(o.pq),
                "mode": o.params.mode,
                "k": o.params.k,
                "ranker": o.ranker,
                "router": o.router,
                "provider": o.decision_provider,
                "engines_asked": o.engines_asked,
                "engines_failed": [n for n, _ in o.engines_failed],
                "n_pool": o.pool_size,
                "urls": [d.url for d in o.docs],
                "warnings": [w.code for w in o.warnings],
                "timings": o.timings,
                "cost": o.usage.get("cost"),
            }
        )


def _short(e: BaseException) -> str:
    """Error summary for warnings: never provider response text (it can echo account identifiers)."""
    if isinstance(e, asyncio.TimeoutError | TimeoutError):
        return "timeout"
    kind = getattr(e, "kind", None)
    if kind:
        status = getattr(e, "status", None)
        return f"{kind}" + (f", HTTP {status}" if status else "")
    return type(e).__name__
