"""Tuning on the dev split: RRF k, BM25F weight, blend α, τ_route; calibration tables per model.

Grades come from the judge; documents never judged count as 0 and the unjudged rate is reported, so a
setting that surfaces many unjudged documents is visible rather than silently penalised.

    uv run python eval/tune.py [--section prior,alpha,route,calibration]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from common import load_queries
from metrics import mean, ndcg_at, unjudged_at
from replay import blend_from, finish, load_run, query_pool
from report import INJECT_PREFIX, grades_by_query

DECISION_ARMS = ["jev", "d1", "mercury", "typesafe", "ollaya"]


def dev_ids() -> list[str]:
    return [e.id for e in load_queries() if e.split == "dev"]


def tune_prior(grades: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    qs = {e.id: e for e in load_queries() if e.split == "dev"}
    rows = []
    for k in (10, 20, 40, 60):
        for w in (0.0, 0.3, 0.5, 0.7, 1.0, 1.5):
            nd, unj = [], []
            for qid, e in qs.items():
                try:
                    pq, docs, _ = query_pool(e, rrf_k=k, bm25_weight=w)
                except FileNotFoundError:
                    continue
                ranking = finish(docs, waive=bool(pq.sites))
                g = {x: v for x, v in grades.get(qid, {}).items() if INJECT_PREFIX not in x}
                nd.append(ndcg_at(ranking, g, 10))
                unj.append(unjudged_at(ranking, g, 10))
            rows.append(
                {
                    "rrf_k": k,
                    "bm25_weight": w,
                    "ndcg10": round(mean(nd), 4),
                    "unjudged10": round(mean(unj), 3),
                }
            )
    rows.sort(key=lambda r: -r["ndcg10"])
    return rows


def tune_alpha(grades: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    qs = {e.id: e for e in load_queries() if e.split == "dev"}
    jev = load_run("jev")
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0):
        nd = []
        for qid, e in qs.items():
            if qid not in jev:
                continue
            ranking = blend_from(jev[qid], e, alpha=alpha)["ranking"]
            g = {x: v for x, v in grades.get(qid, {}).items() if INJECT_PREFIX not in x}
            nd.append(ndcg_at(ranking, g, 10))
        rows.append({"alpha": alpha, "ndcg10": round(mean(nd), 4), "n": len(nd)})
    return rows


def calibration(grades: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Per model: for each p bin, how many judged docs, fraction with grade ≥1, fraction grade 2."""
    out: dict[str, Any] = {}
    for arm in DECISION_ARMS:
        run = load_run(arm)
        if not run:
            continue
        bins: dict[str, list[int]] = defaultdict(list)
        all_p: list[float] = []
        for qid, r in run.items():
            g = grades.get(qid, {})
            for key, p in (r.get("scores") or {}).items():
                if p is None:
                    continue
                all_p.append(p)
                if key in g and INJECT_PREFIX not in key:
                    b = min(int(p * 10), 9)
                    bins[f"{b / 10:.1f}"].append(g[key])
        hist = defaultdict(int)
        for p in all_p:
            hist[f"{min(int(p * 10), 9) / 10:.1f}"] += 1
        out[arm] = {
            "p_histogram": dict(sorted(hist.items())),
            "by_bin": {
                b: {
                    "n": len(v),
                    "rel": round(sum(1 for x in v if x >= 1) / len(v), 2),
                    "high": round(sum(1 for x in v if x == 2) / len(v), 2),
                }
                for b, v in sorted(bins.items())
            },
        }
    return out


def tune_route(grades: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    """Recompute the jev routing arm offline for several route_include thresholds (dev)."""
    from searxng_jev_mcp.decide.providers import Thresholds

    import routing

    rows = []
    for inc in (0.2, 0.35, 0.5, 0.65, 0.8):
        routing.ROUTE_THRESHOLDS = Thresholds(route_include=inc)
        routing.build_arms(load_queries())
        m = routing.routing_metrics(grades, dev_ids())
        keys = ("useful_engine_recall", "relevant_pool_recall", "engines_per_search", "ndcg10", "regret")
        rows.append({"route_include": inc, **{k: m["jev"][k] for k in keys}})
    routing.ROUTE_THRESHOLDS = Thresholds()
    routing.build_arms(load_queries())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="prior,alpha,calibration")
    ap.add_argument("--json", default="eval/private/tune.json")
    a = ap.parse_args()
    grades = grades_by_query()
    out: dict[str, Any] = {}
    sections = a.section.split(",")
    if "prior" in sections:
        out["prior_grid"] = tune_prior(grades)
        print("prior grid (top 8):", json.dumps(out["prior_grid"][:8], indent=0))
    if "alpha" in sections:
        out["alpha_grid"] = tune_alpha(grades)
        print("alpha grid:", json.dumps(out["alpha_grid"], indent=0))
    if "route" in sections:
        out["route_grid"] = tune_route(grades)
        print("route grid:", json.dumps(out["route_grid"], indent=0))
    if "calibration" in sections:
        out["calibration"] = calibration(grades)
        print("calibration:", json.dumps(out["calibration"], indent=1))
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
