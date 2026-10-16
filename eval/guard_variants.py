"""Authority-guard variants, evaluated offline on the stored snapshots (no live searches).

Selection rule (declared before any test-split number was computed; applied to dev only):
  pool     = the production default (general engines + static vertical map, routing off),
             every candidate scored with Jev 1.13 via TypeSafe direct (eval/private/runs/route_e2e_scores)
  N        = condensed nDCG@10 on dev (unjudged documents dropped)
  M        = known-item MRR on dev (16 queries)
  eligible = variants with N >= max(N) - 0.005  (no measurable nDCG cost)
  choose   = the eligible variant with the highest M; if two are within 0.005 of each other on M, the one
             that overrides the reranker least, in the order
             off < pin1_p80 < pin1_band < pin3_p80 < pin3_band < band_reorder < pin3
Implicit-click MRR is reported but not used for selection.

    uv run python eval/guard_variants.py --score        # fill missing candidate scores (TypeSafe direct)
    uv run python eval/guard_variants.py --phase dev    # all variants on dev, applies the rule → selection.json
    uv run python eval/guard_variants.py --phase test   # all variants on test, once (requires selection.json)

Outputs: eval/private/guard_variants_{dev,test}.json, eval/private/guard_variants.md, eval/private/guard_selection.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any

from searxng_jev_mcp import lexical
from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.pipeline import GUARD_MODES, order_by_decision

import replay
import routing_e2e
from common import PRIVATE, doc_key, load_queries
from metrics import mean, ndcg_at, precision_at, reciprocal_rank, unjudged_at
from report import INJECT_PREFIX, grades_by_query, implicit_for, targets_for

POOLS = ("static", "exhaustive")
LEAST_OVERRIDE = ["off", "pin1_p80", "pin1_band", "pin3_p80", "pin3_band", "band_reorder", "pin3"]
SELECTION = PRIVATE / "guard_selection.json"


def rankings(pool: str, guard: str, dedupe: bool = True) -> dict[str, list[str]]:
    decisions = replay.load_run("route_decisions")
    scores = routing_e2e.load_scores()
    out: dict[str, list[str]] = {}
    for e in load_queries():
        pq, docs = routing_e2e.pool_docs(e, pool, decisions, dedupe=dedupe)
        cands = lexical.select_candidates(docs, 60)
        for d in cands:
            p = scores.get(e.id, {}).get(doc_key(d))
            if p is not None and not d.suspect:
                d.p_jev, d.jev_scored = p, True
        ids = {d.id for d in cands}
        ordered = order_by_decision(cands, pq, keep=0.2, guard=guard, general_engines=replay.GENERAL_ENGINES)
        out[e.id] = replay.finish(ordered + [d for d in docs if d.id not in ids], waive=bool(pq.sites))
    return out


def per_query(ranks: dict[str, list[str]], split: str) -> dict[str, dict[str, float]]:
    grades = grades_by_query()
    out: dict[str, dict[str, float]] = {}
    for e in load_queries():
        if e.split != split or e.id not in ranks:
            continue
        g = {k: v for k, v in grades.get(e.id, {}).items() if INJECT_PREFIX not in k}
        r = ranks[e.id]
        m = {
            "ndcg10": ndcg_at(r, g, 10),
            "ndcg10_condensed": ndcg_at([k for k in r if k in g], g, 10),
            "p5": precision_at(r, g, 5),
            "unjudged10": unjudged_at(r, g, 10),
        }
        tg = targets_for(e)
        if tg:
            m["mrr"] = reciprocal_rank(r, tg)
        imp = implicit_for(e)
        if imp:
            m["mrr_implicit"] = reciprocal_rank(r, imp)
        out[e.id] = m
    return out


def _boot_mean(xs: list[float], n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(7)
    stats = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return round(stats[int(0.025 * n)], 4), round(stats[int(0.975 * n)], 4)


def summarize(pq: dict[str, dict[str, float]], base: dict[str, dict[str, float]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("ndcg10", "ndcg10_condensed", "p5", "mrr", "mrr_implicit", "unjudged10"):
        ids = [q for q, m in pq.items() if metric in m]
        if not ids:
            continue
        xs = [pq[q][metric] for q in ids]
        out[metric] = {"mean": round(mean(xs), 4), "ci": _boot_mean(xs), "n": len(xs)}
        if base is not None and metric != "unjudged10":
            diffs = [pq[q][metric] - base[q][metric] for q in ids if q in base and metric in base[q]]
            if diffs:
                out[metric]["vs_off"] = {"diff": round(mean(diffs), 4), "ci": _boot_mean(diffs)}
    return out


def run_phase(split: str) -> dict[str, Any]:
    result: dict[str, Any] = {"split": split, "computed_at": time.time(), "pools": {}}
    for pool in POOLS:
        off = per_query(rankings(pool, "off"), split)
        block = {
            g: summarize(per_query(rankings(pool, g), split) if g != "off" else off, off) for g in GUARD_MODES
        }
        # before/after: the previous default (pin3 guard, no repost dedupe) vs the chosen one (filled at test)
        block["_before_pin3_no_dedupe"] = summarize(
            per_query(rankings(pool, "pin3", dedupe=False), split), off
        )
        result["pools"][pool] = block
    return result


def select(dev: dict[str, Any]) -> dict[str, Any]:
    block = dev["pools"]["static"]
    n = {g: block[g]["ndcg10_condensed"]["mean"] for g in GUARD_MODES}
    m = {g: block[g]["mrr"]["mean"] for g in GUARD_MODES}
    best_n = max(n.values())
    eligible = [g for g in GUARD_MODES if n[g] >= best_n - 0.005]
    top_m = max(m[g] for g in eligible)
    near = [g for g in eligible if top_m - m[g] < 0.005]
    choice = min(near, key=LEAST_OVERRIDE.index)
    rule = (__doc__ or "").split("Selection rule")[1].split("Implicit-click")[0].strip()
    return {"rule": rule, "N": n, "M": m,
            "eligible": eligible, "choice": choice, "decided_at": time.time()}  # fmt: skip


async def score_missing() -> None:
    """Candidates that only appear after repost dedupe need scores (TypeSafe direct)."""
    decisions = replay.load_run("route_decisions")
    scores = routing_e2e.load_scores()
    decider = replay.make_decider("typesafe", load_settings(repo_env_files()))
    n = 0
    for e in load_queries():
        have = scores.setdefault(e.id, {})
        missing = {}
        pq = None
        for pool in POOLS:
            pq, docs = routing_e2e.pool_docs(e, pool, decisions)
            for d in lexical.select_candidates(docs, 60):
                if doc_key(d) not in have and not d.suspect:
                    missing.setdefault(doc_key(d), d)
        if missing and pq is not None:
            batch = list(missing.values())
            await decider.rerank(pq, batch, navigational=False, deadline=60.0)
            for d in batch:
                if d.p_jev is not None:
                    have[doc_key(d)] = d.p_jev
            n += len(batch)
    replay.save_run(routing_e2e.SCORES, scores)
    print(f"scored {n} documents; TypeSafe ${decider.client.spend.total_cost:.4f}")
    await decider.aclose()


def markdown(dev: dict[str, Any], test: dict[str, Any], sel: dict[str, Any]) -> str:
    lines = [f"Selection (dev only): chose **{sel['choice']}**; eligible {sel['eligible']}.", ""]
    for pool in POOLS:
        lines += [f"### Pool: {pool}", "",
                  "| guard | nDCG@10 dev | nDCG@10 test | condensed dev | condensed test | known-item MRR dev | known-item MRR test | implicit MRR dev | implicit MRR test |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for g in [*GUARD_MODES, "_before_pin3_no_dedupe"]:
            row = [g]
            for metric in ("ndcg10", "ndcg10_condensed", "mrr", "mrr_implicit"):
                for blk in (dev, test):
                    v = blk["pools"][pool][g].get(metric)
                    row.append(f"{v['mean']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if v else "—")
            # reorder columns: metric-major (dev, test)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test"])
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score:
        asyncio.run(score_missing())
    if a.phase == "dev":
        dev = run_phase("dev")
        (PRIVATE / "guard_variants_dev.json").write_text(json.dumps(dev, indent=1))
        sel = select(dev)
        SELECTION.write_text(json.dumps(sel, indent=1))
        print(json.dumps({k: sel[k] for k in ("N", "M", "eligible", "choice")}, indent=1))
    elif a.phase == "test":
        sel = json.loads(SELECTION.read_text())  # the choice is fixed before test is computed
        test = run_phase("test")
        (PRIVATE / "guard_variants_test.json").write_text(json.dumps(test, indent=1))
        dev = json.loads((PRIVATE / "guard_variants_dev.json").read_text())
        (PRIVATE / "guard_variants.md").write_text(markdown(dev, test, sel))
        print((PRIVATE / "guard_variants.md").read_text())


if __name__ == "__main__":
    main()
