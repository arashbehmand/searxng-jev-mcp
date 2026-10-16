"""Snapshot: one exhaustive fan-out per eval query, raw SearXNG JSON stored for offline replay.

Per query, two live searches (run in parallel, at most 2 in flight):
  general  – every enabled general engine (bing, yahoo, yandex, wikipedia, duckduckgo, yep, google cse)
  vertical – every enabled vertical engine of the routed categories, with the keyword form of the
             query (what request D sends). Quota-limited engines (Stack Exchange, GitHub) are included
             only for code-like queries; Google Scholar is never included (CAPTCHAs).

    uv run python eval/snapshot.py [--only p01,p02] [--max-searches 200] [--overwrite] [--public-only]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import httpx

from searxng_jev_mcp import query, route
from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.pipeline import vertical_q
from searxng_jev_mcp.searxng import Catalog

from common import SNAPSHOTS, EvalQuery, load_queries, snapshot_path

SNAPSHOT_TIMEOUT_LIMIT = 4.0
GENERAL = ["bing", "yahoo", "yandex", "wikipedia", "duckduckgo", "yep", "google cse"]
QUOTA_CODE_ONLY = {"stackoverflow", "askubuntu", "superuser", "github"}
NEVER = {"google scholar"}
CODE_STRATA = {"code", "operator"}


def vertical_engines(e: EvalQuery, catalog: Catalog) -> list[str]:
    rc = route.load_catalog()
    pq = query.parse_query(e.q)
    codeish = e.stratum in CODE_STRATA or bool(set(route.static_categories(pq)) & {"it", "q&a", "repos"})
    out: list[str] = []
    for cat in route.ROUTED_CATEGORIES:
        if cat == "general":
            continue
        for eng in rc.category_engines.get(cat, []):
            if eng in out or eng in NEVER or eng in GENERAL or not catalog.enabled(eng):
                continue
            # quota engines (one Stack Exchange quota for three sites; GitHub 10/min): code-like
            # queries only, and the engine's keyword gate must fire
            if eng in QUOTA_CODE_ONLY and not (codeish and route.engine_gate_ok(eng, pq)):
                continue
            out.append(eng)
    return out


async def fetch(
    client: httpx.AsyncClient, q: str, engines: list[str], language: str | None
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": q, "format": "json", "engines": ",".join(engines),
        "timeout_limit": f"{SNAPSHOT_TIMEOUT_LIMIT:g}", "locale": "en",
    }  # fmt: skip
    if language:
        params["language"] = language
    t0 = time.perf_counter()
    try:
        r = await client.get("/search", params=params, timeout=SNAPSHOT_TIMEOUT_LIMIT + 2)
        body = r.json() if r.status_code == 200 else None
        status = r.status_code
    except (httpx.HTTPError, ValueError) as ex:
        body, status = None, f"error:{type(ex).__name__}"
    return {"q": q, "engines": engines, "language": language, "status": status, "body": body,
            "elapsed": round(time.perf_counter() - t0, 3)}  # fmt: skip


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--max-searches", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--public-only", action="store_true")
    ap.add_argument("--pause", type=float, default=4.0)
    a = ap.parse_args()
    s = load_settings(repo_env_files())
    assert s.searxng_url
    queries = load_queries(include_private=not a.public_only)
    if a.only:
        wanted = set(a.only.split(","))
        queries = [e for e in queries if e.id in wanted]
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    searches = 0
    async with httpx.AsyncClient(base_url=s.searxng_url, headers={"Accept": "application/json"}) as client:
        catalog = Catalog.from_config((await client.get("/config")).json())
        general = [g for g in GENERAL if catalog.enabled(g)]
        for e in queries:
            path = snapshot_path(e.id)
            if path.exists() and not a.overwrite:
                continue
            if searches + 2 > a.max_searches:
                print(f"stopping: search budget {a.max_searches} reached")
                break
            pq = query.parse_query(e.q)
            verticals = vertical_engines(e, catalog)
            g, v = await asyncio.gather(
                fetch(client, query.to_searxng_q(pq), general, pq.language),
                fetch(client, vertical_q(pq), verticals, pq.language),
            )
            searches += 2
            snap = {"id": e.id, "q": e.q, "ts": time.time(), "searxng_version": catalog.version,
                    "requests": {"general": g, "vertical": v}}  # fmt: skip
            path.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
            ng = len((g["body"] or {}).get("results", []))
            nv = len((v["body"] or {}).get("results", []))
            fails = len((g["body"] or {}).get("unresponsive_engines", [])) + len(
                (v["body"] or {}).get("unresponsive_engines", [])
            )
            print(
                f"{e.id}: general {ng} ({g['elapsed']}s), vertical {nv} ({v['elapsed']}s), unresponsive {fails}"
            )
            await asyncio.sleep(a.pause)
    print(f"live searches used: {searches}")


if __name__ == "__main__":
    asyncio.run(main())
