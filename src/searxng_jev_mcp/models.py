"""Shared data types passed between pipeline stages.

Kept dependency-free (dataclasses only) so every stage module can import it without cycles.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Literal

Script = Literal["latin", "arabic", "cjk", "cyrillic", "other"]


@dataclass(slots=True)
class ParsedQuery:
    """Result of stage A (query.py)."""

    original: str
    # Free text after scrubbing, with operators removed. NOT escaped; query.to_searxng_q() escapes.
    text: str
    # Text used for lexical scoring: free text plus quoted phrases, operators removed, not escaped.
    lexical_text: str
    phrases: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)  # site:example.com (include)
    exclude_sites: list[str] = field(default_factory=list)  # -site:example.com
    exclude_terms: list[str] = field(default_factory=list)  # -term
    filetypes: list[str] = field(default_factory=list)
    script: Script = "latin"
    language: str | None = None  # set only when the script makes it unambiguous
    time_range: str | None = None  # day | week | month | year: only from the caller (cues set recency_cue)
    recency_cue: bool = False  # words like "latest"/"newest": soft recency boost only, no time_range
    scrubbed: list[str] = field(default_factory=list)  # kinds removed, e.g. ["path", "uuid"]
    truncated: bool = False
    raw: bool = False  # caller asked to skip SearXNG-syntax escaping


@dataclass(slots=True)
class RawResult:
    """One row of one SearXNG response (stage C or D), before merging."""

    url: str
    title: str
    content: str
    engines: list[str]
    positions: list[int]
    score: float
    category: str | None = None
    published: dt.datetime | None = None
    request: str = "general"  # which SearXNG request produced it: general | vertical | variant1..3
    rank: int = 0  # 1-based rank by `score` within its request


@dataclass(slots=True)
class Doc:
    """A merged candidate (stage E) carrying ranking features (stages F-G)."""

    id: str  # opaque short id, stable within one search; never derived from rank
    url: str  # URL shown to the caller (https preferred)
    canonical: str  # canonical URL key
    ident: str | None  # identifier key, e.g. "arxiv:2401.01234", "doi:10.1000/x", "gh:owner/repo"
    title: str
    snippet: str
    domain: str  # host without www./m.
    site_key: str  # diversity key: registrable domain, or github.com/<owner>
    engines: set[str] = field(default_factory=set)
    category: str | None = None
    published: dt.datetime | None = None
    request_ranks: dict[str, int] = field(default_factory=dict)  # request label -> best rank
    request_scores: dict[str, float] = field(default_factory=dict)  # request label -> best score
    positions: list[int] = field(default_factory=list)
    from_vertical: bool = False
    # lexical features (stage F)
    bm25: float = 0.0
    bm25_rank: int | None = None
    coverage: float = 0.0  # fraction of distinct query terms present
    phrase_hit: bool = False
    exact_hit: bool = False  # matched a version / error-code style token from the query
    # prior (stage F)
    prior: float = 0.0
    prior_rank: int | None = None
    reserved: str | None = None  # "vertical" | "exact" when it entered the K cut through a reserved slot
    # decisions (stage G)
    p_jev: float | None = None
    jev_scored: bool = False
    suspect: bool = False  # prompt-injection heuristic fired
    final_rank: int | None = None
    flags: list[str] = field(default_factory=list)
    guard_reason: str | None = None  # why the authority guard placed it (explain output)
    rel: Literal["high", "medium", "low"] | None = None  # from the active model's thresholds
    duplicates: list[str] = field(default_factory=list)  # URLs of syndicated reposts merged into this doc
