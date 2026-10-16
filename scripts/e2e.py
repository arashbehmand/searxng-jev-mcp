"""End-to-end checks through a real stdio MCP client against the live SearXNG and decision provider.

Cases: fast, balanced, assess, private, Persian, operator-heavy, SearXNG syntax (`pandas <2`).
About 2 live searches per case. Full outputs go to eval/private/e2e.json (they contain live results);
a summary is printed.

    uv run python scripts/e2e.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import Client

from mcp_client import server_params

CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("fast", "python asyncio gather return_exceptions", {"mode": "fast"}),
    (
        "balanced",
        "how to undo the last git commit but keep the changes",
        {"mode": "balanced", "explain": True},
    ),
    ("assess", "attention is all you need transformer paper", {"mode": "balanced", "assess": True}),
    ("private", "sqlite WAL mode concurrent readers writers", {"mode": "balanced", "private": True}),
    ("persian", "آموزش پایتون برای مبتدیان", {"mode": "balanced", "explain": True}),
    ("operators", '"segmentation fault" core dump gdb -windows site:stackoverflow.com', {"mode": "balanced"}),
    ("searxng_syntax", "pandas <2 DataFrame.append removed !wp :fr", {"mode": "balanced", "explain": True}),
    ("cli_flags", "tar -xvf extract archive to directory", {"mode": "fast", "explain": True}),
    ("year_token", "fastapi lifespan events 2026", {"mode": "fast", "explain": True}),
    ("system_path", "difference between /dev/null and /dev/zero", {"mode": "fast"}),
    ("time_range", "rust release notes", {"mode": "fast", "time_range": "month"}),
]

# Live checks use TypeSafe direct only, with caching off.
ENV = {"SEARXNG_JEV_CACHE": "0", "DECISION_PROVIDER": "typesafe"}


async def main() -> None:
    out: dict[str, Any] = {}
    async with Client(server_params(ENV), read_timeout_seconds=60) as client:
        tools = [t.name for t in (await client.list_tools()).tools]
        print("tools:", tools)
        for name, q, args in CASES:
            res = await client.call_tool("search", {"query": q, "k": 6, **args})
            sc = res.structured_content or {}
            text = res.content[0].text if res.content else ""  # type: ignore[union-attr]
            out[name] = {"args": args, "is_error": res.is_error, "structured": sc, "text": text}
            ex = sc.get("explain") or {}
            print(
                f"{name:15} error={res.is_error} n={len(sc.get('results', []))} ranker={sc.get('ranker')} "
                f"router={sc.get('router')} verdict={sc.get('coverage', {}).get('verdict')} "
                f"total_s={ex.get('timings_s', {}).get('total')} lang={ex.get('language')} "
                f"sent_q={sc.get('query')!r} warnings={[w['code'] for w in sc.get('warnings', [])]}"
            )
    path = Path(__file__).resolve().parent.parent / "eval" / "private" / "e2e.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
