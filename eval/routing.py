"""Routing eval: the ranker is held constant (prior), the vertical engine subset varies.

Arms: general_only, static, hint (the query's category hint as `categories=`), jev (production: the
decision model decides; general dropped when it says so), jev_union (decision routing ∪ static map, the
original rule), exhaustive.

Subsets are simulated from the exhaustive vertical snapshot. Because SearXNG's `positions` are not
per-engine (see docs/verified-facts.md), a row kept for an engine subset gets each kept engine's position
approximated by the row's mean position, and its score recomputed as n'² / mean_position.

    uv run python eval/routing.py --decide        # live routing calls (cheap), stored in runs/route_decisions.json
    uv run python eval/routing.py                 # build arm rankings → runs/route_<arm>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from typing import Any

from searxng_jev_mcp import route
from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.decide.backend import select_route
from searxng_jev_mcp.decide.providers import Thresholds
from searxng_jev_mcp.health import HealthTable
from searxng_jev_mcp.models import ParsedQuery, RawResult
from searxng_jev_mcp.pipeline import build_pool
from searxng_jev_mcp.searxng import Catalog, EngineInfo, SearxResponse
from searxng_jev_mcp.store import Store

from common import RUNS, EvalQuery, doc_key, load_queries, load_snapshot, parsed, snapshot_responses
from replay import finish, inject_doc, load_run, make_decider, save_run

# jev = production: the decision model decides the verticals (static map only if routing fails);
# jev_union = the original rule (decision routing ∪ static map, quota vetoes)
ARMS = ["general_only", "static", "hint", "jev", "jev_union", "exhaustive"]
ROUTE_THRESHOLDS = Thresholds()  # tune.py varies route_include


def snapshot_catalog(responses: dict[str, SearxResponse]) -> Catalog:
    rc = route.load_catalog()
    engines: dict[str, EngineInfo] = {}
    for r in responses.values():
        for e in r.engines_asked:
            cat = rc.engines.get(e, {}).get("category", "general")
            cats = [c for c, es in rc.category_engines.items() if e in es] or [cat]
            engines[e] = EngineInfo(name=e, categories=cats, enabled=True)
    return Catalog(engines=engines)


def available_verticals(pq: ParsedQuery, cat: Catalog, health: HealthTable) -> dict[str, list[str]]:
    from searxng_jev_mcp.decide.questions import ENGINE_QUESTIONS

    out = route.available_verticals(
        pq, cat, health, route.load_catalog(), asked_engines=frozenset(ENGINE_QUESTIONS)
    )
    return {
        c: [e for e in es if e in cat.engines] for c, es in out.items() if any(e in cat.engines for e in es)
    }


def subset_response(resp: SearxResponse, keep: set[str]) -> SearxResponse:
    rows: list[RawResult] = []
    for r in resp.results:
        kept = [e for e in r.engines if e in keep]
        if not kept:
            continue
        mean_pos = statistics.mean(r.positions) if r.positions else 10.0
        n = len(kept)
        rows.append(RawResult(url=r.url, title=r.title, content=r.content, engines=kept, positions=[round(mean_pos)] * n,
                              score=n * n / mean_pos, category=r.category, published=r.published,
                              request=r.request))  # fmt: skip
    rows.sort(key=lambda x: -x.score)
    for i, x in enumerate(rows, start=1):
        x.rank = i
    return SearxResponse(resp.label, resp.query_seen, rows, resp.answers, resp.infoboxes, resp.suggestions,
                         resp.corrections, resp.unresponsive, sorted(keep & set(resp.engines_asked)), resp.elapsed)  # fmt: skip


def arm_engines(arm: str, e: EvalQuery, pq: ParsedQuery, cat: Catalog, health: HealthTable,
                decision: dict[str, Any] | None, all_vertical: set[str]) -> tuple[set[str], bool]:  # fmt: skip
    """(vertical engines, keep_general)."""
    rc = route.load_catalog()
    static = (
        set(route.engines_for_categories(route.static_categories(pq), pq, cat, health, rc)) & all_vertical
    )
    if arm == "general_only":
        return set(), True
    if arm == "exhaustive":
        return set(all_vertical), True
    if arm == "static":
        return static, True
    if arm == "hint":
        if not e.hint or e.hint == "general":
            return set(), True
        return set(
            route.engines_for_categories([e.hint], pq, cat, health, rc, gated=False)
        ) & all_vertical, True
    if decision is None:
        return static, True  # fail open, as production does
    probs = decision["probabilities"]
    available = available_verticals(pq, cat, health)
    trust = 1.0 if pq.script == "latin" else 0.7
    _cats, engines, keep_general = select_route(probs, available, ROUTE_THRESHOLDS, rc, trust=trust)
    jev = set(engines) & all_vertical
    if arm == "jev":
        return jev, keep_general
    vetoed = {
        k.removeprefix("eng:")
        for k, p in probs.items()
        if k.startswith("eng:") and p < ROUTE_THRESHOLDS.engine_include * trust
    }
    static_kept = {x for x in static if not (rc.is_quota(x) and x in vetoed)}
    return (jev | static_kept), keep_general


async def decide_all(queries: list[EvalQuery], force: bool) -> None:
    data = load_run("route_decisions")
    decider = make_decider("jev", load_settings(repo_env_files()))
    health = HealthTable(Store(":memory:"))
    for e in queries:
        if e.id in data and not force:
            continue
        snap = load_snapshot(e.id)
        if not snap:
            continue
        responses = snapshot_responses(snap)
        pq = parsed(e)
        avail = available_verticals(pq, snapshot_catalog(responses), health)
        try:
            rd = await decider.route(pq, avail, 30.0)
            data[e.id] = {"probabilities": rd.probabilities, "categories": rd.categories, "engines": rd.engines,
                          "keep_general": rd.keep_general, "latency_s": decider.client.last_latency}  # fmt: skip
        except Exception as ex:
            print(e.id, "routing failed:", ex)
    save_run("route_decisions", data)
    print(f"routing calls cost ${decider.client.spend.total_cost:.4f}")
    await decider.aclose()


def build_arms(queries: list[EvalQuery]) -> None:
    decisions = load_run("route_decisions")
    health = HealthTable(Store(":memory:"))
    runs: dict[str, dict[str, Any]] = {a: {} for a in ARMS}
    for e in queries:
        snap = load_snapshot(e.id)
        if not snap:
            continue
        pq = parsed(e)
        base = snapshot_responses(snap)
        if e.inject:
            inject_doc(e, base)
        cat = snapshot_catalog(base)
        all_vertical = set(base["vertical"].engines_asked) if "vertical" in base else set()
        for arm in ARMS:
            engines, keep_general = arm_engines(arm, e, pq, cat, health, decisions.get(e.id), all_vertical)
            responses = []
            if "vertical" in base and engines:
                responses.append(subset_response(base["vertical"], engines))
            has_vertical = any(r.results for r in responses)
            if "general" in base and (keep_general or not has_vertical):
                responses.append(base["general"])
            docs = build_pool(pq, responses, recency=pq.recency_cue)
            runs[arm][e.id] = {
                "ranking": finish(docs, waive=bool(pq.sites)),
                "pool": [doc_key(d) for d in docs],
                "engines": sorted(engines),
                "keep_general": keep_general,
                "vertical_providers": {doc_key(d): sorted(d.engines & all_vertical) for d in docs},
                "in_general": [doc_key(d) for d in docs if "general" in d.request_ranks],
            }
    for arm, data in runs.items():
        save_run(f"route_{arm}", data)
    print(f"built routing arms for {len(runs['exhaustive'])} queries → {RUNS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    queries = load_queries()
    if a.decide:
        asyncio.run(decide_all(queries, a.force))
    build_arms(queries)


def routing_metrics(grades: dict[str, dict[str, int]], ids: list[str]) -> dict[str, Any]:
    """Relevant-pool recall, useful-engine recall, engines per search, regret (vs exhaustive)."""
    from metrics import mean, ndcg_at

    runs = {a: load_run(f"route_{a}") for a in ARMS}
    out: dict[str, Any] = {}
    exh = runs["exhaustive"]
    for arm in ARMS:
        recalls, uer_hits, uer_total, n_eng, regrets, ndcgs = [], 0, 0, [], [], []
        for qid in ids:
            if qid not in exh or qid not in runs[arm]:
                continue
            g = {k: v for k, v in grades.get(qid, {}).items() if "seo-spam.example/" not in k}
            rel = {k for k in exh[qid]["pool"] if g.get(k, 0) >= 1}
            pool = set(runs[arm][qid]["pool"])
            if rel:
                recalls.append(len(rel & pool) / len(rel))
            in_general = set(exh[qid]["in_general"])
            useful: set[str] = set()
            for k in rel - in_general:
                useful |= set(exh[qid]["vertical_providers"].get(k, []))
            uer_total += len(useful)
            uer_hits += len(useful & set(runs[arm][qid]["engines"]))
            n_eng.append(len(runs[arm][qid]["engines"]))
            nd = ndcg_at(runs[arm][qid]["ranking"], g, 10)
            ndcgs.append(nd)
            regrets.append(ndcg_at(exh[qid]["ranking"], g, 10) - nd)
        out[arm] = {
            "relevant_pool_recall": round(mean(recalls), 4) if recalls else None,
            "useful_engine_recall": round(uer_hits / uer_total, 4) if uer_total else None,
            "useful_engines_total": uer_total,
            "engines_per_search": round(mean(n_eng), 2) if n_eng else None,
            "ndcg10": round(mean(ndcgs), 4) if ndcgs else None,
            "regret": round(mean(regrets), 4) if regrets else None,
            "n": len(ndcgs),
        }
    return out


def print_routing(grades: dict[str, dict[str, int]], ids: list[str]) -> None:
    m = routing_metrics(grades, ids)
    print(json.dumps(m, indent=1))


def useful_engine_ci(
    grades: dict[str, dict[str, int]], ids: list[str], base: str, arm: str, n: int = 10_000
) -> dict[str, float]:
    """Paired query-bootstrap CI for the useful-engine-recall difference (ratio estimator) arm − base."""
    import random

    runs = {a: load_run(f"route_{a}") for a in (base, arm, "exhaustive")}
    rows = []
    for qid in ids:
        exh = runs["exhaustive"].get(qid)
        if not exh or qid not in runs[base] or qid not in runs[arm]:
            continue
        g = grades.get(qid, {})
        rel = {k for k in exh["pool"] if g.get(k, 0) >= 1 and "seo-spam.example/" not in k}
        useful: set[str] = set()
        for k in rel - set(exh["in_general"]):
            useful |= set(exh["vertical_providers"].get(k, []))
        rows.append(
            (
                len(useful),
                len(useful & set(runs[base][qid]["engines"])),
                len(useful & set(runs[arm][qid]["engines"])),
            )
        )

    def stat(rs: list[tuple[int, int, int]]) -> float:
        tot = sum(r[0] for r in rs)
        return (sum(r[2] for r in rs) - sum(r[1] for r in rs)) / tot if tot else 0.0

    rng = random.Random(7)
    boots = sorted(stat([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return {"diff": round(stat(rows), 4), "lo": round(boots[int(0.025 * n)], 4), "hi": round(boots[int(0.975 * n)], 4),
            "queries_with_useful_engines": sum(1 for r in rows if r[0])}  # fmt: skip


if __name__ == "__main__":
    main()
