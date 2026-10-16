"""SearXNG HTTP client: `/config` catalog, `/search` with `timeout_limit`, recording and tracing.

Facts this relies on (verified on SearXNG 2026.10, see docs/verified-facts.md):
- `engines=` names engines even if they are disabled, so callers filter with :class:`Catalog`.
- `timeout_limit` makes SearXNG return partial results and list slow engines in
  `unresponsive_engines` as `[name, reason]`.
- JSON array order is regrouped by category, so ranks are computed from `score`.
- With `format=json` disabled SearXNG answers 403.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .models import RawResult

USER_AGENT = "searxng-jev-mcp/0.1"


class SearxngError(Exception):
    """Hard failure talking to SearXNG. `kind` is one of: unreachable, json_disabled, rate_limited, http, bad_response."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(slots=True)
class EngineInfo:
    name: str
    categories: list[str]
    enabled: bool
    paging: bool = False
    time_range_support: bool = False
    language_support: bool = False
    timeout: float | None = None
    shortcut: str | None = None


@dataclass(slots=True)
class Catalog:
    engines: dict[str, EngineInfo]
    version: str | None = None
    fetched_at: float = field(default_factory=time.time)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Catalog:
        engines: dict[str, EngineInfo] = {}
        for e in cfg.get("engines", []):
            name = e.get("name")
            if not name:
                continue
            engines[name] = EngineInfo(
                name=name,
                categories=list(e.get("categories") or []),
                enabled=bool(e.get("enabled")),
                paging=bool(e.get("paging")),
                time_range_support=bool(e.get("time_range_support")),
                language_support=bool(e.get("language_support")),
                timeout=e.get("timeout"),
                shortcut=e.get("shortcut"),
            )
        return cls(engines=engines, version=cfg.get("version"))

    def enabled(self, name: str) -> bool:
        info = self.engines.get(name)
        return bool(info and info.enabled)

    def in_category(self, category: str) -> list[str]:
        return [n for n, e in self.engines.items() if e.enabled and category in e.categories]

    def categories(self) -> list[str]:
        return sorted({c for e in self.engines.values() if e.enabled for c in e.categories})


@dataclass(slots=True)
class SearxResponse:
    label: str
    query_seen: str
    results: list[RawResult]
    answers: list[str]
    infoboxes: list[dict[str, Any]]
    suggestions: list[str]
    corrections: list[str]
    unresponsive: list[tuple[str, str]]
    engines_asked: list[str]
    elapsed: float
    from_cache: bool = False
    redirected: bool = False

    @property
    def engines_failed(self) -> set[str]:
        return {name for name, _ in self.unresponsive}


def parse_published(value: Any) -> dt.datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        d = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=dt.UTC)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def parse_response(
    data: dict[str, Any], *, label: str, engines_asked: list[str], elapsed: float
) -> SearxResponse:
    """Turn SearXNG's JSON into a :class:`SearxResponse`; ranks come from `score`, not array order."""
    if not isinstance(data, dict) or "results" not in data:
        raise SearxngError("bad_response", "SearXNG returned JSON without a `results` field")
    rows: list[RawResult] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        engines = [e for e in item.get("engines") or [item.get("engine")] if isinstance(e, str) and e]
        positions = [int(p) for p in item.get("positions") or [] if isinstance(p, int | float)]
        rows.append(
            RawResult(
                url=str(item["url"]),
                title=_text(item.get("title")),
                content=_text(item.get("content")),
                engines=engines,
                positions=positions,
                score=float(item.get("score") or 0.0),
                category=item.get("category") if isinstance(item.get("category"), str) else None,
                published=parse_published(item.get("publishedDate")),
                request=label,
            )
        )
    order = sorted(range(len(rows)), key=lambda i: -rows[i].score)  # stable for ties
    for rank, i in enumerate(order, start=1):
        rows[i].rank = rank
    rows.sort(key=lambda r: r.rank)

    answers: list[str] = []
    for a in data.get("answers") or []:
        if isinstance(a, dict):
            text = a.get("answer") or a.get("content") or a.get("text")
            if isinstance(text, str) and text.strip():
                url = a.get("url")
                answers.append(f"{text.strip()} ({url})" if isinstance(url, str) and url else text.strip())
        elif isinstance(a, str) and a.strip():
            answers.append(a.strip())
    infoboxes = [
        {
            "title": _text(ib.get("infobox")),
            "content": _text(ib.get("content"))[:600],
            "url": _text(ib.get("id") or ib.get("url")),
            "engine": _text(ib.get("engine")),
        }
        for ib in data.get("infoboxes") or []
        if isinstance(ib, dict)
    ]
    unresponsive: list[tuple[str, str]] = []
    for pair in data.get("unresponsive_engines") or []:
        if isinstance(pair, list | tuple) and pair:
            unresponsive.append((str(pair[0]), str(pair[1]) if len(pair) > 1 else ""))
    return SearxResponse(
        label=label,
        query_seen=_text(data.get("query")),
        results=rows,
        answers=answers,
        infoboxes=infoboxes,
        suggestions=[s for s in data.get("suggestions") or [] if isinstance(s, str)],
        corrections=[s for s in data.get("corrections") or [] if isinstance(s, str)],
        unresponsive=unresponsive,
        engines_asked=list(engines_asked),
        elapsed=elapsed,
    )


class Recorder:
    """Saves raw SearXNG responses (request params + JSON) for snapshots and fixtures."""

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def save(self, params: dict[str, Any], status: int, body: Any, elapsed: float) -> Path:
        digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        path = self.directory / f"{digest}.json"
        path.write_text(
            json.dumps(
                {"params": params, "status": status, "elapsed": elapsed, "body": body}, ensure_ascii=False
            )
        )
        return path


class SearxngClient:
    """Async SearXNG client with a concurrency cap (the instance is small; agent bursts get engines banned)."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: tuple[str, str] | None = None,
        max_concurrency: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        recorder: Recorder | None = None,
        locale: str = "en",
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            auth=auth,
            transport=transport,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            follow_redirects=False,
        )
        self._sem = asyncio.Semaphore(max_concurrency)
        self._catalog: Catalog | None = None
        self._catalog_lock = asyncio.Lock()
        self.recorder = recorder
        self.locale = locale
        self.live_searches = 0  # count of /search requests actually sent (not cache hits)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def catalog(self, *, max_age: float = 6 * 3600) -> Catalog:
        async with self._catalog_lock:
            if self._catalog and time.time() - self._catalog.fetched_at < max_age:
                return self._catalog
            data = await self._get_json("/config", {}, timeout=10.0)
            self._catalog = Catalog.from_config(data)
            return self._catalog

    async def search(
        self,
        q: str,
        *,
        engines: list[str],
        label: str = "general",
        language: str | None = None,
        time_range: str | None = None,
        pageno: int = 1,
        timeout_limit: float = 2.5,
        safesearch: int | None = None,
    ) -> SearxResponse:
        params: dict[str, Any] = {
            "q": q,
            "format": "json",
            "engines": ",".join(engines),
            "timeout_limit": f"{timeout_limit:g}",
            "pageno": pageno,
            "locale": self.locale,
        }
        if language:
            params["language"] = language
        if time_range:
            params["time_range"] = time_range
        if safesearch is not None:
            params["safesearch"] = safesearch
        async with self._sem:
            t0 = time.perf_counter()
            try:
                resp = await self._client.get("/search", params=params, timeout=timeout_limit + 0.5)
            except httpx.TimeoutException as e:
                raise SearxngError(
                    "unreachable", f"SearXNG did not answer within {timeout_limit + 0.5:.1f}s"
                ) from e
            except httpx.HTTPError as e:
                raise SearxngError("unreachable", f"cannot reach SearXNG: {type(e).__name__}") from e
            finally:
                self.live_searches += 1
            elapsed = time.perf_counter() - t0
        if resp.status_code in (301, 302, 303, 307, 308):
            # A bang or `!!` redirect (only possible with raw=true): there are no results to parse.
            return SearxResponse(label, q, [], [], [], [], [], [], list(engines), elapsed, redirected=True)
        body = self._check(resp)
        if self.recorder:
            self.recorder.save(params, resp.status_code, body, elapsed)
        return parse_response(body, label=label, engines_asked=engines, elapsed=elapsed)

    async def _get_json(self, path: str, params: dict[str, Any], timeout: float) -> Any:  # noqa: ASYNC109
        try:
            resp = await self._client.get(path, params=params, timeout=timeout)
        except httpx.HTTPError as e:
            raise SearxngError("unreachable", f"cannot reach SearXNG: {type(e).__name__}") from e
        return self._check(resp)

    @staticmethod
    def _check(resp: httpx.Response) -> Any:
        if resp.status_code == 403:
            raise SearxngError(
                "json_disabled",
                "SearXNG answered 403: add `json` to `search.formats` in settings.yml (see README).",
            )
        if resp.status_code == 429:
            raise SearxngError(
                "rate_limited", "SearXNG answered 429: disable the limiter or whitelist this client."
            )
        if resp.status_code >= 400:
            raise SearxngError("http", f"SearXNG answered HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as e:
            raise SearxngError("bad_response", "SearXNG did not return JSON") from e
