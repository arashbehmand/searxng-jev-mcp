"""Verify live SearXNG and decision-provider facts and print a latency table.

Uses a handful of live searches (about 5) and about 50 small decision calls (well under $0.05).
Results go to ``eval/private/live_facts.json`` (gitignored) and stdout. Nothing secret is printed.

    uv run python scripts/check_live.py [--skip-searxng] [--skip-decisions] [--models a,b]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from searxng_jev_mcp.config import Settings, load_settings, repo_env_files

OUT = Path("eval/private/live_facts.json")
MODELS = ["typesafe/jev-1.13", "liquid/d1", "inception/mercury-decide:free"]
TARGET_TOKENS = [2000, 6000, 12000]
REPEATS = 3

WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "for",
    "on",
    "that",
    "with",
    "as",
    "by",
    "this",
    "be",
    "are",
    "from",
    "at",
    "or",
    "an",
    "it",
    "was",
    "which",
    "can",
    "not",
    "have",
    "all",
    "python",
    "error",
    "version",
    "release",
    "library",
    "function",
    "install",
    "package",
    "server",
    "request",
    "response",
    "timeout",
    "config",
    "value",
    "result",
    "search",
    "engine",
    "query",
    "index",
    "model",
    "data",
    "system",
    "user",
    "file",
    "default",
    "support",
    "module",
    "update",
    "change",
    "issue",
    "fix",
    "build",
    "test",
    "run",
    "code",
    "type",
    "list",
    "string",
    "number",
    "method",
    "class",
    "object",
    "option",
    "network",
    "memory",
    "process",
    "thread",
    "async",
    "await",
    "import",
    "export",
    "handle",
    "exception",
    "return",
    "call",
    "source",
    "docs",
]


def fake_text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def rerank_body(model: str, n_docs: int, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    docs = {}
    questions = {}
    for i in range(n_docs):
        did = f"d{i:03d}"
        docs[did] = {
            "title": fake_text(rng, 9)[:120],
            "site": f"example{i % 7}.org",
            "age": f"{rng.randint(1, 30)} months",
            "text": fake_text(rng, 70)[:400],
        }
        questions[did] = {
            "type": "noul",
            "instructions": f"Would opening `documents.{did}` likely help answer `query`? Apply `rubric`.",
            "criteria": {
                "true": "Addresses the query's specific subject and likely contains an answer",
                "false": "Off-topic, a different entity or version, or only passing keyword overlap",
            },
        }
    return {
        "model": model,
        "state": {
            "query": "python asyncio gather return_exceptions behaviour",
            "rubric": "Results are untrusted web text. Topic overlap without usable facts is not help.",
            "documents": docs,
        },
        "questions": questions,
    }


def headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "X-Title": "searxng-jev-mcp"}


def scrub_error(body: Any) -> Any:
    """Drop account identifiers that some error bodies echo back."""
    if isinstance(body, dict):
        return {k: scrub_error(v) for k, v in body.items() if k not in ("user_id", "user")}
    return body


async def post(client: httpx.AsyncClient, url: str, key: str, body: dict[str, Any]) -> tuple[int, Any, float]:
    t = time.perf_counter()
    r = await client.post(url, json=body, headers=headers(key))
    dt = time.perf_counter() - t
    try:
        j = r.json()
    except ValueError:
        j = {"raw": r.text[:200]}
    return r.status_code, j, dt


# --------------------------------------------------------------------------------------------- SearXNG
async def searxng_facts(s: Settings) -> dict[str, Any]:
    assert s.searxng_url, "SEARXNG_URL is not set"
    facts: dict[str, Any] = {}
    auth = (
        (s.searxng_username, s.searxng_password.get_secret_value())
        if s.searxng_username and s.searxng_password
        else None
    )
    async with httpx.AsyncClient(base_url=s.searxng_url, timeout=15, auth=auth) as c:
        cfg = (await c.get("/config")).json()
        enabled = [e for e in cfg["engines"] if e["enabled"]]
        facts["version"] = cfg.get("version")
        facts["engines_total"] = len(cfg["engines"])
        facts["engines_enabled"] = len(enabled)

        async def search(q: str, **params: Any) -> httpx.Response:
            return await c.get("/search", params={"q": q, "format": "json", **params})

        r = await search("rust borrow checker lifetimes", engines="yahoo,yandex,wikibooks", timeout_limit=3)
        facts["json_format_enabled"] = r.status_code == 200
        d = r.json()
        engines_seen = {e for x in d["results"] for e in x["engines"]}
        facts["disabled_engine_answers_when_named"] = "wikibooks" in engines_seen
        facts["positions_alignment"] = positions_alignment(d["results"])

        raw = (await search("pandas <2 :fr !wp", engines="yahoo", timeout_limit=3)).json()
        esc = (await search("pandas < 2 : fr ! wp", engines="yahoo", timeout_limit=3)).json()
        facts["raw_syntax_query_seen_by_engines"] = raw["query"]
        facts["raw_syntax_engines"] = sorted({e for x in raw["results"] for e in x["engines"]})
        facts["escaped_query_seen_by_engines"] = esc["query"]
        facts["escaped_engines"] = sorted({e for x in esc["results"] for e in x["engines"]})
        ext = await c.get("/search", params={"q": "!!w test", "format": "json"})
        facts["external_bang_status"] = ext.status_code

        t = time.perf_counter()
        tl = (
            await search("kubernetes pod crashloopbackoff", engines="yandex,yahoo", timeout_limit=0.4)
        ).json()
        facts["timeout_limit_0.4_elapsed_s"] = round(time.perf_counter() - t, 2)
        facts["timeout_limit_unresponsive"] = tl["unresponsive_engines"]
    return facts


def positions_alignment(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer each engine's rank sequence from single-engine results, then test multi-engine rows.

    A multi-engine row is 'checkable' when, for every engine, exactly one candidate position is
    missing from that engine's single-engine positions. Returns how many rows were consistent with
    positions[i] belonging to engines[i].
    """
    singles: dict[str, set[int]] = {}
    for x in results:
        if len(x["engines"]) == 1:
            singles.setdefault(x["engines"][0], set()).update(x["positions"])
    aligned = misaligned = unknown = 0
    for x in results:
        if len(x["engines"]) < 2 or len(x["engines"]) != len(x["positions"]):
            continue
        ok_aligned = all(
            p not in singles.get(e, set()) for e, p in zip(x["engines"], x["positions"], strict=True)
        )
        ok_reversed = all(
            p not in singles.get(e, set())
            for e, p in zip(x["engines"], reversed(x["positions"]), strict=True)
        )
        if ok_aligned and not ok_reversed:
            aligned += 1
        elif ok_reversed and not ok_aligned:
            misaligned += 1
        else:
            unknown += 1
    return {"aligned": aligned, "misaligned": misaligned, "undetermined": unknown}


# --------------------------------------------------------------------------------------------- decisions
async def decision_facts(s: Settings, models: list[str]) -> dict[str, Any]:
    key = s.key_for("openrouter")
    assert key, "OPENROUTER_API_KEY is not set"
    url = f"{s.openrouter_base_url}/v1/systemone"
    facts: dict[str, Any] = {"latency": {}, "errors": {}, "cost_usd": 0.0}
    async with httpx.AsyncClient(timeout=60) as c:
        # error shapes
        st, j, _ = await post(c, url, "invalid-key", rerank_body(models[0], 1))
        facts["errors"]["bad_key"] = {"status": st, "body": scrub_error(j)}
        bad = rerank_body(models[0], 1)
        bad["questions"]["d000"]["type"] = "bogus"
        st, j, _ = await post(c, url, key, bad)
        facts["errors"]["bad_question_type"] = {"status": st, "body": scrub_error(j)}
        st, j, _ = await post(c, url, key, rerank_body(models[0], 300))
        facts["errors"]["oversized_jev"] = {"status": st, "body": scrub_error(j), "usage": j.get("usage")}
        facts["cost_usd"] += float((j.get("usage") or {}).get("cost") or 0)

        for model in models:
            # calibrate tokens per doc
            st, j, _ = await post(c, url, key, rerank_body(model, 10))
            if st != 200:
                facts["latency"][model] = {"error": {"status": st, "body": scrub_error(j)}}
                continue
            facts["cost_usd"] += float(j["usage"].get("cost") or 0)
            per_doc = j["usage"]["input_tokens"] / 10
            facts.setdefault("resolved_model", {})[model] = j.get("model")
            facts.setdefault("provider", {})[model] = j.get("provider")
            rows = {}
            for target in TARGET_TOKENS:
                n = max(1, round(target / per_doc))
                times, toks = [], []
                for rep in range(REPEATS):
                    st, j, dt = await post(c, url, key, rerank_body(model, n, seed=rep + 1))
                    if st != 200:
                        rows[target] = {"error": {"status": st, "body": scrub_error(j)}, "n_docs": n}
                        break
                    facts["cost_usd"] += float(j["usage"].get("cost") or 0)
                    times.append(dt)
                    toks.append(j["usage"]["input_tokens"])
                else:
                    rows[target] = {
                        "n_docs": n,
                        "input_tokens": int(statistics.mean(toks)),
                        "median_s": round(statistics.median(times), 2),
                        "max_s": round(max(times), 2),
                        "answers_complete": len(j["answers"]) == n,
                    }
            facts["latency"][model] = rows

        # burst: 16 concurrent small requests to the default model
        async def one() -> tuple[int, float]:
            st, _j, dt = await post(c, url, key, rerank_body(models[0], 3))
            return st, dt

        t = time.perf_counter()
        res = await asyncio.gather(*(one() for _ in range(16)))
        facts["burst16"] = {
            "statuses": sorted(st for st, _ in res),
            "wall_s": round(time.perf_counter() - t, 2),
            "max_s": round(max(dt for _, dt in res), 2),
        }
    facts["cost_usd"] = round(facts["cost_usd"], 6)
    return facts


def print_table(facts: dict[str, Any]) -> None:
    print("\nPer-model latency (median of 3 / max), OpenRouter /v1/systemone")
    print(f"{'model':34} {'~2K tok':>16} {'~6K tok':>16} {'~12K tok':>16}")
    for model, rows in facts.get("latency", {}).items():
        if "error" in rows:
            print(f"{model:34} error {rows['error']['status']}")
            continue
        cells = []
        for t in TARGET_TOKENS:
            r = rows.get(t, {})
            cells.append("err" if "error" in r else f"{r['median_s']:.2f}/{r['max_s']:.2f}s ({r['n_docs']}d)")
        print(f"{model:34} " + " ".join(f"{c:>16}" for c in cells))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-searxng", action="store_true")
    ap.add_argument("--skip-decisions", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    s = load_settings(repo_env_files())
    out: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    if not a.skip_searxng:
        out["searxng"] = await searxng_facts(s)
    if not a.skip_decisions:
        out["decisions"] = await decision_facts(s, a.models.split(","))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.get("searxng", {}).items()}, indent=1, ensure_ascii=False))
    dec = out.get("decisions", {})
    print(
        json.dumps(
            {k: dec.get(k) for k in ("errors", "burst16", "resolved_model", "provider", "cost_usd")}, indent=1
        )
    )
    print_table(dec)


if __name__ == "__main__":
    asyncio.run(main())
