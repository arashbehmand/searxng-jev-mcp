"""End-to-end routing eval: decision rerank on each routed pool vs decision rerank on the general-only pool.

The routing eval in routing.py holds the *prior* constant; this one holds the *decision rerank*
constant, which is what balanced mode actually does. Every candidate document of every pool variant
gets a Jev score. Scores come from the stored TypeSafe-direct arm (same Jev 1.13 as OpenRouter,
r = 0.988) and missing documents are scored with TypeSafe direct (`--score`, a few cents).

    uv run python eval/routing_e2e.py --score     # fill missing scores (TypeSafe direct)
    uv run python eval/routing_e2e.py             # metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from typing import Any

from searxng_jev_mcp import lexical
from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.health import HealthTable
from searxng_jev_mcp.models import Doc, ParsedQuery
from searxng_jev_mcp.pipeline import build_pool, order_by_decision
from searxng_jev_mcp.store import Store

import replay
import routing
from common import EvalQuery, doc_key, load_queries, load_snapshot, parsed, snapshot_responses
from metrics import mean, ndcg_at, precision_at, reciprocal_rank, unjudged_at
from report import INJECT_PREFIX, grades_by_query, targets_for

POOLS = ["general_only", "static", "jev", "jev_union", "exhaustive"]
SCORES = "route_e2e_scores"


def pool_docs(
    e: EvalQuery, arm: str, decisions: dict[str, Any], *, dedupe: bool = True
) -> tuple[ParsedQuery, list[Doc]]:
    snap = load_snapshot(e.id)
    assert snap
    base = snapshot_responses(snap)
    if e.inject:
        replay.inject_doc(e, base)
    pq = parsed(e)
    cat = routing.snapshot_catalog(base)
    allv = set(base["vertical"].engines_asked) if "vertical" in base else set()
    health = HealthTable(Store(":memory:"))
    engines, keep_general = routing.arm_engines(arm, e, pq, cat, health, decisions.get(e.id), allv)
    responses = []
    if "vertical" in base and engines:
        responses.append(routing.subset_response(base["vertical"], engines))
    has_vertical = any(r.results for r in responses)
    if "general" in base and (keep_general or not has_vertical):
        responses.append(base["general"])
    return pq, build_pool(pq, responses, recency=pq.recency_cue, dedupe=dedupe)


def load_scores() -> dict[str, dict[str, float]]:
    stored = replay.load_run(SCORES)
    if stored:
        return stored
    ts = replay.load_run("typesafe")
    return {qid: dict(r.get("scores") or {}) for qid, r in ts.items()}


async def score_missing() -> None:
    decisions = replay.load_run("route_decisions")
    scores = load_scores()
    decider = replay.make_decider("typesafe", load_settings(repo_env_files()))
    n_docs = 0
    for e in load_queries():
        have = scores.setdefault(e.id, {})
        missing: dict[str, Doc] = {}
        pq: ParsedQuery | None = None
        for arm in POOLS:
            pq, docs = pool_docs(e, arm, decisions)
            for d in lexical.select_candidates(docs, 60):
                k = doc_key(d)
                if k not in have and not d.suspect and k not in missing:
                    missing[k] = d
        if missing and pq is not None:
            batch = list(missing.values())
            info = await decider.rerank(pq, batch, navigational=False, deadline=60.0)
            for d in batch:
                if d.p_jev is not None:
                    have[doc_key(d)] = d.p_jev
            n_docs += len(batch)
            if info.get("error"):
                print(e.id, "error", info["error"])
    replay.save_run(SCORES, scores)
    print(f"scored {n_docs} new documents; TypeSafe spend ${decider.client.spend.total_cost:.4f}")
    await decider.aclose()


def evaluate() -> dict[str, Any]:
    decisions = replay.load_run("route_decisions")
    scores = load_scores()
    grades = grades_by_query()
    per: dict[str, dict[str, dict[str, float]]] = {a: {} for a in POOLS}
    meta: dict[str, dict[str, Any]] = {a: {"engines": []} for a in POOLS}
    for e in load_queries():
        g = {k: v for k, v in grades.get(e.id, {}).items() if INJECT_PREFIX not in k}
        tg = targets_for(e)
        for arm in POOLS:
            pq, docs = pool_docs(e, arm, decisions)
            cands = lexical.select_candidates(docs, 60)
            for d in cands:
                p = scores.get(e.id, {}).get(doc_key(d))
                if p is not None and not d.suspect:
                    d.p_jev, d.jev_scored = p, True
            ids = {d.id for d in cands}
            ordered = order_by_decision(cands, pq, keep=0.2, general_engines=replay.GENERAL_ENGINES)
            ranking = replay.finish(ordered + [d for d in docs if d.id not in ids], waive=bool(pq.sites))
            judged = [k for k in ranking if k in g]
            m = {
                "ndcg10": ndcg_at(ranking, g, 10),
                "ndcg10_condensed": ndcg_at(judged, g, 10),
                "p5": precision_at(ranking, g, 5),
                "unjudged10": unjudged_at(ranking, g, 10),
            }
            if tg:
                m["mrr"] = reciprocal_rank(ranking, tg)
            per[arm][e.id] = m
            engines, _ = routing.arm_engines(
                arm, e, pq, routing.snapshot_catalog(snapshot_responses(load_snapshot(e.id) or {})),
                HealthTable(Store(":memory:")), decisions.get(e.id),
                set((load_snapshot(e.id) or {}).get("requests", {}).get("vertical", {}).get("engines", [])),
            )  # fmt: skip
            meta[arm]["engines"].append(len(engines))
    queries = load_queries()
    out: dict[str, Any] = {}
    for split in ("dev", "test", "all"):
        ids = [e.id for e in queries if split == "all" or e.split == split]
        block: dict[str, Any] = {}
        for arm in POOLS:
            rows = [per[arm][i] for i in ids if i in per[arm]]
            block[arm] = {k: round(mean([r[k] for r in rows if k in r]), 4) for k in
                          ("ndcg10", "ndcg10_condensed", "p5", "unjudged10", "mrr")}  # fmt: skip
            block[arm]["vertical_engines"] = round(mean(meta[arm]["engines"]), 2)
            if arm != "general_only":
                for metric in ("ndcg10_condensed", "ndcg10"):
                    a = [per["general_only"][i][metric] for i in ids]
                    b = [per[arm][i][metric] for i in ids]
                    block[arm][f"vs_general_{metric}"] = _boot(a, b)
        out[split] = block
    # per-stratum, all queries, condensed
    strata: dict[str, Any] = {}
    for s in sorted({e.stratum for e in queries}):
        ids = [e.id for e in queries if e.stratum == s]
        strata[s] = {arm: round(mean([per[arm][i]["ndcg10_condensed"] for i in ids]), 3) for arm in POOLS} | {
            "n": len(ids)
        }
    out["by_stratum_condensed"] = strata
    return out


def _boot(a: list[float], b: list[float], n: int = 10_000) -> dict[str, float]:
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(7)
    stats = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(n))
    return {"diff": round(mean(diffs), 4), "lo": round(stats[250], 4), "hi": round(stats[9750], 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--json", default="eval/private/routing_e2e.json")
    a = ap.parse_args()
    if a.score:
        asyncio.run(score_missing())
    out = evaluate()
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for split in ("dev", "test", "all"):
        print(f"== {split}")
        for arm, v in out[split].items():
            vs = v.get("vs_general_ndcg10_condensed")
            print(f"  {arm:13} nDCG {v['ndcg10']:.4f} condensed {v['ndcg10_condensed']:.4f} unjudged {v['unjudged10']:.3f} "
                  f"MRR {v['mrr']:.3f} engines {v['vertical_engines']}" + (f"  vs general (condensed) {vs}" if vs else ""))  # fmt: skip
    print("by stratum (condensed nDCG):", json.dumps(out["by_stratum_condensed"], indent=0))


if __name__ == "__main__":
    main()
