"""Sample the private eval queries from transcript searches (seeded, stratified; no hand-picking).

Skips any query that scrubbing would alter (paths, usernames, emails, keys, IPs): those are personal
and would not reach SearXNG unchanged anyway. Output: eval/private/queries.private.yaml (gitignored).

    uv run python eval/build_private_set.py [--n 43] [--seed 20261008]
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import yaml

from searxng_jev_mcp import query, route

from common import PRIVATE, read_jsonl

QUOTAS = {"code": 0.40, "science": 0.12, "news": 0.12, "howto": 0.16, "i18n": 0.10, "other": 0.10}


def stratum(q: str) -> str:
    pq = query.parse_query(q)
    if pq.script != "latin":
        return "i18n"
    cats = route.static_categories(pq)
    if "science" in cats:
        return "science"
    if "news" in cats or pq.time_range or pq.recency_cue:
        return "news"
    if {"it", "q&a", "repos", "packages"} & set(cats):
        return "code"
    if q.lower().startswith(("how ", "how to", "what is", "why ")):
        return "howto"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=43)
    ap.add_argument("--seed", type=int, default=20261008)
    a = ap.parse_args()
    rows = read_jsonl(PRIVATE / "transcript_queries.jsonl")
    pool: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        q = r["query"]
        if not 8 <= len(q) <= 200:
            continue
        _, kinds = query.scrub(q)
        if kinds:
            continue
        pool[stratum(q)].append(r)
    rng = random.Random(a.seed)
    chosen: list[tuple[str, dict]] = []
    for s, frac in QUOTAS.items():
        items = pool.get(s, [])
        rng.shuffle(items)
        chosen += [(s, r) for r in items[: max(1, round(a.n * frac))]]
    chosen = chosen[: a.n]
    # fill any shortfall (empty strata) from the remaining pool, still seeded
    taken = {id(r) for _, r in chosen}
    rest = [(s, r) for s, items in pool.items() for r in items if id(r) not in taken]
    rng.shuffle(rest)
    chosen += rest[: a.n - len(chosen)]
    out = {"queries": [
        {"id": f"x{i:02d}", "q": r["query"], "stratum": s, "fetched": r["fetched"][:3]}
        for i, (s, r) in enumerate(chosen, start=1)
    ]}  # fmt: skip
    path = PRIVATE / "queries.private.yaml"
    path.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    counts = defaultdict(int)
    for s, _ in chosen:
        counts[s] += 1
    print(f"pool sizes: { {k: len(v) for k, v in pool.items()} }")
    print(
        f"chosen {len(chosen)}: {dict(counts)}, with fetches: {sum(1 for _, r in chosen if r['fetched'])} → {path}"
    )


if __name__ == "__main__":
    main()
