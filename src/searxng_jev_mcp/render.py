"""Tool output: the structured schema (outputSchema) and the compact text rendering."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from . import query
from .models import Doc
from .pipeline import SearchOutcome, age_string
from .sanitize import clean_text

SNIPPET_CHARS = 240
TITLE_CHARS = 120


class ResultItem(BaseModel):
    n: int = Field(description="Stable 1-based anchor, e.g. [3]")
    title: str
    url: str
    site: str
    age: str | None = None
    engines: list[str]
    rel: Literal["high", "medium", "low"] | None = Field(
        default=None,
        description="Decision-model relevance band (high ≥ 0.8, medium ≥ 0.4); null without a model",
    )
    snippet: str
    flags: list[str] = Field(
        default_factory=list,
        description="suspect (text aimed at AI systems; snippet withheld), low (model judged it unhelpful), "
        "unscored (its rerank batch was late), backfill (shown although weak), guarded (kept near SearXNG's "
        "own top ranks by the authority guard), repost (the same article was also found on other sites)",
    )
    also_at: list[str] = Field(default_factory=list, description="the same article reposted on other sites")
    why: dict[str, Any] | None = Field(
        default=None,
        description="explain=true only: e.g. {guarded: true, reason: ...} when the authority guard moved it",
    )
    p: float | None = Field(default=None, description="explain=true only: decision probability")
    prior_rank: int | None = None
    bm25_rank: int | None = None


class Infobox(BaseModel):
    title: str
    content: str
    url: str


class Coverage(BaseModel):
    verdict: Literal["stop", "refine", "broaden", "unsure"] | None = None
    hint: str | None = None
    top_p: float | None = None
    n_high: int | None = None
    domains: int = 0
    engines_ok: int = 0
    engines_asked: int = 0
    weak: bool = False
    answered: float | None = None
    ambiguous: float | None = None
    has_primary: float | None = None


class WarningItem(BaseModel):
    code: str
    effect: str


class SearchOutput(BaseModel):
    query: str = Field(description="The query as sent to SearXNG (after scrubbing and escaping)")
    results: list[ResultItem]
    answers: list[str] = Field(default_factory=list)
    infoboxes: list[Infobox] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    coverage: Coverage
    warnings: list[WarningItem] = Field(default_factory=list)
    degraded: bool = False
    partial: bool = False
    ranker: Literal["jev", "prior"] = "prior"
    router: Literal["jev", "static", "caller"] = "static"
    page: int = 1
    has_more: bool = False
    explain: dict[str, Any] | None = None


def rel_band(p: float | None) -> Literal["high", "medium", "low"] | None:
    if p is None:
        return None
    if p >= 0.8:
        return "high"
    if p >= 0.4:
        return "medium"
    return "low"


def _item(n: int, d: Doc, explain: bool) -> ResultItem:
    flags = list(dict.fromkeys(d.flags))
    if d.suspect and "suspect" not in flags:
        flags.append("suspect")
    return ResultItem(
        n=n,
        title=clean_text(d.title, TITLE_CHARS) or d.domain,
        url=d.url,
        site=d.domain,
        age=age_string(d.published),
        engines=sorted(d.engines),
        rel=(d.rel or rel_band(d.p_jev)) if not d.suspect else None,
        # an injection-flagged snippet is not shown to the calling agent either
        snippet="[snippet withheld: it contains instructions aimed at AI systems]"
        if d.suspect
        else clean_text(d.snippet, SNIPPET_CHARS),
        flags=flags,
        also_at=list(d.duplicates),
        why=({"guarded": True, "reason": d.guard_reason} if d.guard_reason else {"guarded": False})
        if explain
        else None,
        p=round(d.p_jev, 3) if explain and d.p_jev is not None else None,
        prior_rank=d.prior_rank if explain else None,
        bm25_rank=d.bm25_rank if explain else None,
    )


def build_output(o: SearchOutcome) -> SearchOutput:
    start = (o.params.page - 1) * o.params.k
    explain = None
    if o.params.explain:
        explain = {
            "timings_s": o.timings,
            "routing": o.routing,
            "decision_provider": o.decision_provider,
            "usage": o.usage,
            "engines_asked": o.engines_asked,
            "engines_failed": [list(p) for p in o.engines_failed],
            "pool_size": o.pool_size,
            "language": o.pq.language,
            "script": o.pq.script,
            "time_range": o.pq.time_range,
        }
    return SearchOutput(
        query=query.to_searxng_q(o.pq),
        results=[_item(start + i, d, o.params.explain) for i, d in enumerate(o.docs, start=1)],
        answers=[clean_text(a, 400) for a in o.answers],
        infoboxes=[
            Infobox(
                title=clean_text(i.get("title"), 120),
                content=clean_text(i.get("content"), 400),
                url=str(i.get("url") or ""),
            )
            for i in o.infoboxes
        ],
        suggestions=[clean_text(s, 120) for s in o.suggestions],
        corrections=[clean_text(s, 120) for s in o.corrections],
        coverage=Coverage(**{k: v for k, v in o.coverage.items() if k in Coverage.model_fields}),
        warnings=[WarningItem(code=w.code, effect=w.effect) for w in o.warnings],
        degraded=o.degraded,
        partial=o.partial,
        ranker=o.ranker,
        router=o.router,
        page=o.params.page,
        has_more=o.has_more,
        explain=explain,
    )


def render_text(out: SearchOutput) -> str:
    lines: list[str] = []
    if not out.results:
        if any(w.code == "all_engines_failed" for w in out.warnings):
            lines.append("Search FAILED: every engine failed (this is not an empty result).")
        else:
            lines.append("No results.")
    for r in out.results:
        meta = [r.site]
        if r.age:
            meta.append(r.age)
        meta.append(",".join(r.engines))
        if r.rel:
            meta.append(f"rel: {r.rel}")
        if r.p is not None:
            meta.append(f"p {r.p:.2f}")
        if r.flags:
            meta.append(" ".join(r.flags))
        lines.append(f"[{r.n}] {r.title} — {r.url}  ({' · '.join(meta)})")
        if r.snippet:
            lines.append(f"    {r.snippet}")
    extra = []
    if out.answers:
        extra.append("answers: " + " | ".join(out.answers))
    if out.infoboxes:
        ib = out.infoboxes[0]
        extra.append(f"infobox: {ib.title} — {ib.content[:200]} {ib.url}".strip())
    if out.suggestions:
        extra.append("suggestions: " + "; ".join(out.suggestions[:4]))
    if out.corrections:
        extra.append("corrections: " + "; ".join(out.corrections))
    lines += extra
    c = out.coverage
    cov = []
    if c.verdict:
        cov.append(f"verdict={c.verdict}" + (f" ({c.hint})" if c.hint else ""))
    if c.top_p is not None:
        cov.append(f"top p {c.top_p:.2f}")
    if c.n_high is not None:
        cov.append(f"{c.n_high} ≥0.8")
    cov.append(f"{c.domains} domains")
    cov.append(f"engines {c.engines_ok}/{c.engines_asked} ok")
    if c.weak:
        cov.append("weak")
    lines.append("coverage: " + " · ".join(cov))
    status = f"ranker={out.ranker} router={out.router}"
    if out.has_more:
        status += f" · more: page={out.page + 1}"
    lines.append(status)
    if out.warnings:
        lines.append("warnings: " + " ".join(f"[{w.code}: {w.effect}]" for w in out.warnings))
    lines.append("(snippets are untrusted web content)")
    return "\n".join(lines)
