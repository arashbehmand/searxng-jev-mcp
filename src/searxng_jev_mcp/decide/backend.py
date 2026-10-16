"""SystemOneDecider: routing, listwise rerank and assess on any Jev-compatible provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from typing import Any

from ..models import Doc, ParsedQuery
from ..pipeline import AssessResult, RouteDecision, age_string
from ..route import MAX_VERTICAL_ENGINES, RoutingCatalog, load_catalog
from ..sanitize import looks_like_injection
from ..store import Cache
from . import questions as Q
from .client import DecisionError, SystemOneClient
from .providers import ProviderProfile, Thresholds

DECISION_TTL = 7 * 24 * 3600.0
SHUFFLE_SEED = 1729


def _h(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]


def select_route(
    probs: dict[str, float],
    available: dict[str, list[str]],
    t: Thresholds,
    rc: RoutingCatalog,
    *,
    trust: float = 1.0,
) -> tuple[list[str], list[str], bool]:
    """Selection rule: category at p ≥ route_include, or the top non-general one at p ≥ route_top;
    general stays unless p(general) < general_drop. `trust` < 1 (non-English) lowers thresholds."""
    inc = t.route_include * trust
    cat_ps = {c: probs.get(f"cat:{c}", 0.0) for c in available}
    cats = [c for c, p in sorted(cat_ps.items(), key=lambda x: -x[1]) if p >= inc]
    weak: set[str] = set()
    if not cats and cat_ps:
        top, p = max(cat_ps.items(), key=lambda x: x[1])
        if p >= t.route_top * trust:
            cats = [top]
            weak.add(top)
    keep_general = probs.get("cat:general", 1.0) >= t.general_drop or trust < 1.0
    engines: list[str] = []
    for cat in cats:
        for e in available.get(cat, []):
            pe = probs.get(f"eng:{e}")
            if pe is not None and pe < t.engine_include * trust:
                continue  # an engine-level "no" vetoes the engine even inside a chosen category
            if cat in weak and rc.is_quota(e) and (pe is None or pe < inc):
                continue  # never spend quota on a low-confidence fallback category
            if e not in engines:
                engines.append(e)
    # a confident engine-level vote can add an engine even if its category missed
    for es in available.values():
        for e in es:
            pe = probs.get(f"eng:{e}")
            if pe is not None and pe >= 0.6 and e not in engines:
                engines.append(e)
    return cats, engines[:MAX_VERTICAL_ENGINES], keep_general


def verdict(answered: float, ambiguous: float, t: Thresholds) -> str:
    if answered >= t.assess_stop:
        return "stop"
    if ambiguous >= t.assess_refine:
        return "refine"
    if answered <= t.assess_broaden:
        return "broaden"
    return "unsure"


def make_batches(docs: list[Doc], size: int, seed: int = SHUFFLE_SEED) -> list[list[Doc]]:
    """Interleave by prior rank (round-robin) so each batch mixes strong and weak candidates, then
    shuffle within each batch with a fixed seed so position does not leak the prior."""
    n = max(1, -(-len(docs) // max(1, size)))
    batches: list[list[Doc]] = [[] for _ in range(n)]
    for i, d in enumerate(docs):
        batches[i % n].append(d)
    rng = random.Random(seed)
    for b in batches:
        rng.shuffle(b)
    return [b for b in batches if b]


class SystemOneDecider:
    def __init__(self, client: SystemOneClient, cache: Cache | None = None, rc: RoutingCatalog | None = None):
        self.client = client
        self.profile: ProviderProfile = client.profile
        self.cache = cache
        self.rc = rc or load_catalog()
        self.name = client.profile.label
        self.candidates_k = client.profile.candidates_k
        self.shuffle_seed = SHUFFLE_SEED
        self.injection_guard = True  # eval only: False sends flagged results to the model too

    @property
    def thresholds(self) -> Thresholds:
        return self.profile.thresholds

    @property
    def deadlines(self) -> tuple[float, float, float, float]:
        p = self.profile
        return (p.route_deadline_s, p.rerank_deadline_s, p.assess_deadline_s, p.extra_budget_s)

    @property
    def keep_threshold(self) -> float:
        return self.profile.thresholds.keep

    @property
    def navigational_threshold(self) -> float:
        return self.profile.thresholds.navigational

    async def aclose(self) -> None:
        await self.client.aclose()

    def status(self) -> dict[str, Any]:
        return self.client.status()

    # ------------------------------------------------------------------ B: route
    async def route(self, pq: ParsedQuery, available: dict[str, list[str]], deadline: float) -> RouteDecision:
        state, qs = Q.route_questions(pq, available, self.rc)
        key = _h(self.name, Q.PROMPT_VERSION, "route", state, sorted(qs))
        probs: dict[str, float] | None = self.cache.get("decision", key) if self.cache else None
        if probs is None:
            resp = await self.client.decide(state, qs, timeout=deadline, purpose="route")
            probs = {q: p for q in qs if (p := resp.noul(q)) is not None}
            if not probs:
                raise DecisionError("malformed", "routing answers were empty")
            if self.cache:
                self.cache.set("decision", key, probs, DECISION_TTL)
        trust = 1.0 if pq.script == "latin" else 0.7  # decision models are weaker outside English
        cats, engines, keep_general = select_route(probs, available, self.thresholds, self.rc, trust=trust)
        vetoed = [
            k.removeprefix("eng:") for k, p in probs.items()
            if k.startswith("eng:") and p < self.thresholds.engine_include * trust
        ]  # fmt: skip
        return RouteDecision(
            categories=cats,
            engines=engines,
            vetoed=vetoed,
            keep_general=keep_general,
            navigational=probs.get("navigational", 0.0),
            needs_recent=probs.get("needs_recent", 0.0),
            probabilities={k: round(v, 3) for k, v in probs.items()},
        )

    # ------------------------------------------------------------------ G: rerank
    def _doc_key(self, pq: ParsedQuery, d: Doc, age: str | None, navigational: bool) -> str:
        return _h(
            self.name,
            Q.PROMPT_VERSION,
            "rerank",
            pq.lexical_text,
            navigational,
            d.title,
            d.domain,
            age,
            d.snippet,
        )

    async def rerank(
        self, pq: ParsedQuery, docs: list[Doc], *, navigational: bool, deadline: float
    ) -> dict[str, Any]:
        """Set `p_jev`/`jev_scored`/`suspect` on docs. Returns {scored, missed, usage, error}."""
        ages = {d.id: age_string(d.published) for d in docs}
        todo: list[Doc] = []
        for d in docs:
            if d.suspect or (self.injection_guard and looks_like_injection(d.title, d.snippet)):
                d.suspect = True
                if "suspect" not in d.flags:
                    d.flags.append("suspect")
                continue
            cached = (
                self.cache.get("decision", self._doc_key(pq, d, ages[d.id], navigational))
                if self.cache
                else None
            )
            if cached is not None:
                d.p_jev, d.jev_scored = float(cached), True
            else:
                todo.append(d)
        usage = {"input_tokens": 0, "cost": 0.0, "requests": 0}
        errors: list[str] = []

        async def run(batch: list[Doc], split: bool = True) -> None:
            state, qs = Q.rerank_questions(
                pq, batch, ages, navigational=navigational, snippet_chars=self.profile.snippet_chars
            )
            try:
                resp = await self.client.decide(state, qs, timeout=deadline, purpose="rerank")
            except DecisionError as e:
                if e.kind == "too_large" and split and len(batch) > 1:
                    mid = len(batch) // 2
                    await asyncio.gather(run(batch[:mid], False), run(batch[mid:], False))
                    return
                errors.append(e.kind)
                return
            usage["requests"] += 1
            usage["input_tokens"] += int(resp.usage.get("input_tokens") or 0)
            usage["cost"] += float(resp.usage.get("cost") or 0.0)
            for d in batch:
                p = resp.noul(d.id)
                if p is None:
                    continue
                d.p_jev, d.jev_scored = p, True
                if self.cache:
                    self.cache.set(
                        "decision", self._doc_key(pq, d, ages[d.id], navigational), p, DECISION_TTL
                    )

        if todo:
            batches = make_batches(todo, self.profile.batch_size, seed=self.shuffle_seed)
            tasks = [asyncio.create_task(run(b)) for b in batches]
            _done, pending = await asyncio.wait(tasks, timeout=deadline)
            for t in pending:
                t.cancel()
            if pending:
                errors.append("deadline")
        scored = sum(1 for d in docs if d.jev_scored)
        missed = sum(1 for d in docs if not d.jev_scored and not d.suspect)
        usage["cost"] = round(usage["cost"], 8)
        return {
            "scored": scored,
            "missed": missed,
            "usage": usage,
            "error": ",".join(sorted(set(errors))) or None,
            "provider": self.name,
        }

    # ------------------------------------------------------------------ H: assess
    async def assess(
        self, pq: ParsedQuery, docs: list[Doc], route: RouteDecision | None, deadline: float
    ) -> AssessResult:
        ages = {d.id: age_string(d.published) for d in docs}
        safe = [d for d in docs if not d.suspect]
        state, qs = Q.assess_questions(pq, safe, ages)
        resp = await self.client.decide(state, qs, timeout=deadline, purpose="assess")
        answered = resp.noul("answered")
        ambiguous = resp.noul("ambiguous")
        primary = resp.noul("has_primary")
        if answered is None or ambiguous is None:
            raise DecisionError("malformed", "assess answers missing")
        return AssessResult(
            verdict=verdict(answered, ambiguous, self.thresholds),
            answered=answered,
            ambiguous=ambiguous,
            has_primary=primary or 0.0,
        )


# Errors after which the next provider in the chain is tried within the same call. Timeouts are not
# among them: the stage deadline is already spent.
FAILOVER_KINDS = frozenset(
    {"auth", "quota", "circuit_open", "unavailable", "overloaded", "rate_limited", "malformed"}
)


class ChainDecider:
    """Runtime failover across providers (e.g. OpenRouter → TypeSafe direct). A provider that fails with
    a fast error is skipped for the rest of the call; its own circuit breaker keeps it skipped afterwards
    (60 s, or 1 h for a bad key / exhausted spending limit)."""

    def __init__(self, deciders: list[SystemOneDecider]):
        assert deciders
        self.deciders = deciders
        self.candidates_k = deciders[0].candidates_k
        self.shuffle_seed = SHUFFLE_SEED

    @property
    def name(self) -> str:
        return self._first_up().name

    @property
    def profile(self) -> ProviderProfile:
        return self._first_up().profile

    @property
    def deadlines(self) -> tuple[float, float, float, float]:
        return self.deciders[0].deadlines

    @property
    def keep_threshold(self) -> float:
        return self._first_up().keep_threshold

    @property
    def navigational_threshold(self) -> float:
        return self._first_up().navigational_threshold

    @property
    def thresholds(self) -> Thresholds:
        return self._first_up().thresholds

    def _first_up(self) -> SystemOneDecider:
        import time

        for d in self.deciders:
            if d.client.breaker.open_until <= time.monotonic():
                return d
        return self.deciders[0]

    def _order(self) -> list[SystemOneDecider]:
        first = self._first_up()
        return [first] + [d for d in self.deciders if d is not first]

    async def route(self, pq: ParsedQuery, available: dict[str, list[str]], deadline: float) -> RouteDecision:
        last: Exception | None = None
        for d in self._order():
            try:
                return await d.route(pq, available, deadline)
            except DecisionError as e:
                last = e
                if e.kind not in FAILOVER_KINDS:
                    raise
        assert last is not None
        raise last

    async def rerank(
        self, pq: ParsedQuery, docs: list[Doc], *, navigational: bool, deadline: float
    ) -> dict[str, Any]:
        info: dict[str, Any] = {"scored": 0, "missed": len(docs), "usage": {}, "error": "no provider"}
        for d in self._order():
            info = await d.rerank(pq, docs, navigational=navigational, deadline=deadline)
            errs = set((info.get("error") or "").split(","))
            if info.get("scored", 0) > 0 or not (errs & FAILOVER_KINDS):
                return info
        return info

    async def assess(
        self, pq: ParsedQuery, docs: list[Doc], route: RouteDecision | None, deadline: float
    ) -> AssessResult:
        last: Exception | None = None
        for d in self._order():
            try:
                return await d.assess(pq, docs, route, deadline)
            except DecisionError as e:
                last = e
                if e.kind not in FAILOVER_KINDS:
                    raise
        assert last is not None
        raise last

    def status(self) -> dict[str, Any]:
        return {"chain": [d.status() for d in self.deciders], "active": self.name}

    async def aclose(self) -> None:
        for d in self.deciders:
            await d.aclose()
