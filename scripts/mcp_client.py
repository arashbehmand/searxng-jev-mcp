"""Minimal stdio MCP client: launches the server as a subprocess and calls `search`.

    uv run python scripts/mcp_client.py "python asyncio gather" --mode fast --k 5
    uv run python scripts/mcp_client.py "query" --args '{"assess": true, "explain": true}' --json

The server is started with --env-file for the repo's .env and .env.local (when they exist); it never
reads .env files from the working directory on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def server_params(extra_env: dict[str, str] | None = None) -> StdioServerParameters:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ("HOME", "PATH", "USER", "LANG", "TMPDIR", "XDG_CACHE_HOME")
    }
    env.update(extra_env or {})
    # The server reads no env files implicitly; this dev client names the repo's files explicitly.
    files = [a for f in (".env", ".env.local") if (ROOT / f).exists() for a in ("--env-file", str(ROOT / f))]
    return StdioServerParameters(
        command=sys.executable, args=["-m", "searxng_jev_mcp.server", *files], env=env, cwd=str(ROOT)
    )


async def call(
    query: str, args: dict[str, object], extra_env: dict[str, str] | None = None
) -> tuple[object, float]:
    async with Client(server_params(extra_env), read_timeout_seconds=60) as client:
        t0 = time.perf_counter()
        res = await client.call_tool("search", {"query": query, **args})
        return res, time.perf_counter() - t0


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--args", default="{}")
    ap.add_argument("--json", action="store_true", help="print structured content")
    ap.add_argument("--env", default="", help="extra env, e.g. SEARXNG_JEV_CACHE=0,DECISION_PROVIDER=none")
    a = ap.parse_args()
    extra = dict(kv.split("=", 1) for kv in a.env.split(",") if "=" in kv)
    args = {"mode": a.mode, "k": a.k, **json.loads(a.args)}
    res, dt = await call(a.query, args, extra)
    print(f"--- is_error={res.is_error} wall={dt:.2f}s")  # type: ignore[attr-defined]
    for block in res.content:  # type: ignore[attr-defined]
        print(getattr(block, "text", block))
    if a.json:
        print(json.dumps(res.structured_content, indent=1, ensure_ascii=False))  # type: ignore[attr-defined]


if __name__ == "__main__":
    asyncio.run(main())
