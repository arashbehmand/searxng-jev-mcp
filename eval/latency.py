"""Live latency through the real pipeline: cache off, fresh health state, randomized order.

Modes: fast; balanced (decision rerank, static routing: the default); balanced_routing (decision
routing on); assess (balanced + assess). Decision calls go to the provider in DECISION_PROVIDER (for
the published run: TypeSafe direct). Each (query, mode) pair costs 1-2 live searches.

    DECISION_PROVIDER=typesafe uv run python eval/latency.py --n 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.pipeline import SearchParams
from searxng_jev_mcp.server import build_app

from common import PRIVATE, load_queries
from metrics import percentile

BUDGET = {
    "fast": (1.5, 3.0, 4.0),
    "balanced": (2.3, 5.0, 6.0),
    "balanced_routing": (2.3, 5.0, 6.0),
    "assess": (2.9, 5.6, 6.5),
}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--modes", default="fast,balanced,balanced_routing")
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    fresh = Path(tempfile.mkdtemp(prefix="sjm-latency-"))  # cold health table, no cache
    s = load_settings(repo_env_files()).model_copy(
        update={"cache_enabled": False, "trace_enabled": False, "data_dir": fresh}
    )
    app = build_app(s)
    qs = load_queries(include_private=False)
    picked = []
    for stratum in ("code", "science", "news", "howto", "nav", "operator", "i18n"):
        picked += [e for e in qs if e.stratum == stratum][:2]
    picked = picked[: a.n]
    pairs = [(e, m) for e in picked for m in a.modes.split(",")]
    random.Random(a.seed).shuffle(pairs)
    rows: list[dict[str, Any]] = []
    before = app.searx.live_searches
    for e, mode in pairs:
        app.pipeline.decision_routing = mode == "balanced_routing"
        params = SearchParams(e.q, mode="fast" if mode == "fast" else "balanced", assess=mode == "assess")
        t0 = time.perf_counter()
        try:
            o = await app.pipeline.search(params)
            rows.append({"id": e.id, "mode": mode, "total": time.perf_counter() - t0, "timings": o.timings,
                         "ranker": o.ranker, "router": o.router, "provider": o.decision_provider,
                         "warnings": [w.code for w in o.warnings], "cost": o.usage.get("cost")})  # fmt: skip
        except Exception as ex:
            rows.append({"id": e.id, "mode": mode, "error": f"{type(ex).__name__}: {str(ex)[:120]}"})
        await asyncio.sleep(a.pause)
    summary: dict[str, Any] = {}
    for mode in a.modes.split(","):
        ts = [r["total"] for r in rows if r["mode"] == mode and "total" in r]
        if not ts:
            continue
        p50, p95 = percentile(ts, 0.5), percentile(ts, 0.95)
        target = BUDGET[mode]
        summary[mode] = {"n": len(ts), "p50": round(p50, 2), "p95": round(p95, 2), "max": round(max(ts), 2),
                         "target_p50_p95_hard": target, "p95_within_budget": p95 <= target[1],
                         "decision_ranked": sum(1 for r in rows if r["mode"] == mode and r.get("ranker") == "jev"),
                         "errors": sum(1 for r in rows if r["mode"] == mode and "error" in r)}  # fmt: skip
        for stage in ("route", "searxng", "rerank", "assess"):
            st = [r["timings"][stage] for r in rows if r["mode"] == mode and stage in r.get("timings", {})]
            if st:
                summary[mode][f"{stage}_p50"] = round(percentile(st, 0.5), 2)
                summary[mode][f"{stage}_p95"] = round(percentile(st, 0.95), 2)
    searches = app.searx.live_searches - before
    cost = sum(r.get("cost") or 0 for r in rows)
    out = {"summary": summary, "rows": rows, "live_searches": searches, "decision_cost_usd": round(cost, 5),
           "provider": s.decision_provider}  # fmt: skip
    (PRIVATE / "latency.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    print(f"live searches used: {searches}; decision cost ${cost:.4f}")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
