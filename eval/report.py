"""Metrics per arm and split, paired bootstrap CIs, and the adoption bar.

Adoption bar, written down before any results: a decision stage is on by default if its dev
gain is ≥ 0.05 nDCG@10 (≥ 10% useful-engine recall for routing) with a CI excluding 0 and p95 latency
within budget, AND it beats the cross-encoder arm. Otherwise it ships available but off by default.

    uv run python eval/report.py --arms searxng,prior_nobm25,prior,ce,jev,jev_blend [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from common import JUDGMENTS, PRIVATE, EvalQuery, load_queries, read_jsonl, url_key
from metrics import mean, ndcg_at, paired_bootstrap, precision_at, reciprocal_rank, unjudged_at
from replay import load_run, query_pool

INJECT_PREFIX = "seo-spam.example/"
CONDENSED = (
    False  # --condensed: drop unjudged documents from rankings before scoring (Sakai's condensed lists)
)


def grades_by_query() -> dict[str, dict[str, int]]:
    g: dict[str, dict[str, int]] = defaultdict(dict)
    for j in read_jsonl(JUDGMENTS):
        if j.get("grade") is not None:
            g[j["qid"]][j["key"]] = int(j["grade"])
    return g


def targets_for(e: EvalQuery) -> set[str]:
    """Doc keys in the query's pool that match its known-item targets."""
    from common import target_match

    if not e.targets:
        return set()
    try:
        _pq, docs, _ = query_pool(e)
    except FileNotFoundError:
        return set()
    return {(d.ident or d.canonical) for d in docs if target_match(d, e.targets)}


def implicit_for(e: EvalQuery) -> set[str]:
    return {url_key(u) for u in e.fetched}


def per_query(arms: list[str], queries: list[EvalQuery]) -> dict[str, dict[str, dict[str, float]]]:
    grades = grades_by_query()
    runs = {a: load_run(a) for a in arms}
    out: dict[str, dict[str, dict[str, float]]] = {a: {} for a in arms}
    for e in queries:
        g = {k: v for k, v in grades.get(e.id, {}).items() if INJECT_PREFIX not in k}
        tg = targets_for(e)
        imp = implicit_for(e)
        for a in arms:
            r = runs[a].get(e.id)
            if not r:
                continue
            ranking = r["ranking"]
            if CONDENSED:
                ranking = [k for k in ranking if k in g]
            m: dict[str, float] = {
                "ndcg10": ndcg_at(ranking, g, 10),
                "p5": precision_at(ranking, g, 5),
                "unjudged10": unjudged_at([k for k in ranking if INJECT_PREFIX not in k], g, 10),
            }
            if tg:
                m["mrr"] = reciprocal_rank(ranking, tg)
            if imp:
                m["mrr_implicit"] = reciprocal_rank(ranking, imp)
            if e.inject:
                pos = next((i for i, k in enumerate(ranking, start=1) if INJECT_PREFIX in k), None)
                m["inject_top3"] = float(pos is not None and pos <= 3)
                m["inject_top10"] = float(pos is not None and pos <= 10)
            out[a][e.id] = m
    return out


def summarize(pq: dict[str, dict[str, dict[str, float]]], arms: list[str], ids: list[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for a in arms:
        rows = [pq[a][i] for i in ids if i in pq[a]]
        cols: dict[str, Any] = {"n": len(rows)}
        for metric in ("ndcg10", "p5", "unjudged10", "mrr", "mrr_implicit", "inject_top3", "inject_top10"):
            vals = [r[metric] for r in rows if metric in r]
            if vals:
                cols[metric] = round(mean(vals), 4)
                cols[f"n_{metric}"] = len(vals)
        table[a] = cols
    return table


def compare(pq: dict[str, Any], a: str, b: str, ids: list[str], metric: str = "ndcg10") -> dict[str, float]:
    common_ids = [i for i in ids if i in pq[a] and i in pq[b] and metric in pq[a][i] and metric in pq[b][i]]
    res = paired_bootstrap([pq[a][i][metric] for i in common_ids], [pq[b][i][metric] for i in common_ids])
    return {k: round(v, 4) for k, v in res.items()} | {"n": len(common_ids)}


def adoption(pq: dict[str, Any], stage_arm: str, dev_ids: list[str], p95_ok: bool | None) -> dict[str, Any]:
    vs_prior = compare(pq, "prior", stage_arm, dev_ids)
    vs_ce = compare(pq, "ce", stage_arm, dev_ids) if "ce" in pq else None
    gain_ok = vs_prior["diff"] >= 0.05 and vs_prior["lo"] > 0
    beats_ce = bool(vs_ce and vs_ce["diff"] > 0)
    return {
        "arm": stage_arm,
        "dev_gain_vs_prior": vs_prior,
        "dev_vs_ce": vs_ce,
        "gain_ok": gain_ok,
        "beats_ce": beats_ce,
        "p95_ok": p95_ok,
        "adopt": bool(gain_ok and beats_ce and p95_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--stage-arms", default="jev")
    ap.add_argument("--p95-ok", default="", help="arm=1/0 pairs from the latency run, e.g. jev=1")
    ap.add_argument("--json", default="")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--condensed", action="store_true")
    a = ap.parse_args()
    global CONDENSED
    CONDENSED = a.condensed
    arms = a.arms.split(",")
    queries = load_queries()
    pq = per_query(arms, queries)
    by_split = {s: [e.id for e in queries if e.split == s] for s in ("dev", "test")}
    strata = sorted({e.stratum for e in queries})
    out: dict[str, Any] = {
        "dev": summarize(pq, arms, by_split["dev"]),
        "test": summarize(pq, arms, by_split["test"]),
        "all": summarize(pq, arms, [e.id for e in queries]),
        "public": summarize(pq, arms, [e.id for e in queries if not e.private]),
        "private": summarize(pq, arms, [e.id for e in queries if e.private]),
        "by_stratum": {s: summarize(pq, arms, [e.id for e in queries if e.stratum == s]) for s in strata},
        "pairs_dev": {
            f"{x}->{y}": compare(pq, x, y, by_split["dev"])
            for x in arms
            for y in arms
            if x in ("prior", "ce", "searxng") and y != x
        },
        "pairs_test": {
            f"{x}->{y}": compare(pq, x, y, by_split["test"])
            for x in arms
            for y in arms
            if x in ("prior", "ce") and y != x
        },
    }
    p95 = {k: v == "1" for k, v in (kv.split("=") for kv in a.p95_ok.split(",") if "=" in kv)}
    lat_path = PRIVATE / "latency.json"
    if lat_path.exists():  # balanced-mode p95 from the live latency run applies to every decision arm
        lat = json.loads(lat_path.read_text())["summary"].get("balanced", {})
        for s_arm in a.stage_arms.split(","):
            p95.setdefault(s_arm, bool(lat.get("p95_within_budget")))
    if "prior" in arms:
        out["adoption"] = {
            s: adoption(pq, s, by_split["dev"], p95.get(s)) for s in a.stage_arms.split(",") if s in arms
        }
    out["stability"] = stability()
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    if a.markdown:
        for split in ("dev", "test", "all"):
            print(f"\n### {split}\n")
            print(markdown_table(out[split], arms))
    for split in ("dev", "test", "all"):
        print(f"\n== {split}")
        print(
            f"{'arm':14} {'n':>3} {'nDCG@10':>8} {'P@5':>6} {'MRR(n)':>11} {'impl(n)':>11} {'unj@10':>7} {'inj@3':>6}"
        )
        for arm, c in out[split].items():
            mrr = f"{c.get('mrr', float('nan')):.3f}({c.get('n_mrr', 0)})"
            imp = f"{c.get('mrr_implicit', float('nan')):.3f}({c.get('n_mrr_implicit', 0)})"
            print(f"{arm:14} {c['n']:>3} {c.get('ndcg10', float('nan')):>8.4f} {c.get('p5', float('nan')):>6.3f} "
                  f"{mrr:>11} {imp:>11} {c.get('unjudged10', float('nan')):>7.3f} {c.get('inject_top3', float('nan')):>6.2f}")  # fmt: skip
    print("\n== paired dev nDCG@10 (diff = right − left, 95% CI)")
    for k, v in out["pairs_dev"].items():
        print(f"{k:24} {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] n={v['n']}")
    if "adoption" in out:
        print("\n== adoption bar (dev)")
        for s, v in out["adoption"].items():
            print(s, json.dumps({k: v[k] for k in ("gain_ok", "beats_ce", "p95_ok", "adopt")}))


def stability(arms: tuple[str, ...] = ("jev", "jev_s1", "jev_s2", "jev_s3")) -> dict[str, Any]:
    """Per-document score correlation between Jev runs with permuted batches (bar r ≥ 0.9)."""
    from itertools import combinations

    from metrics import pearson

    runs = {a: load_run(a) for a in arms}
    out: dict[str, Any] = {}
    for a, b in combinations(arms, 2):
        xs, ys = [], []
        for qid, r in runs[a].items():
            other = runs[b].get(qid)
            if not other:
                continue
            for key, p in (r.get("scores") or {}).items():
                q = (other.get("scores") or {}).get(key)
                if p is not None and q is not None:
                    xs.append(p)
                    ys.append(q)
        diffs = [abs(x - y) for x, y in zip(xs, ys, strict=True)]
        out[f"{a}~{b}"] = {"n_docs": len(xs), "pearson": round(pearson(xs, ys), 4),
                           "mean_abs_diff": round(mean(diffs), 4) if diffs else None,
                           "max_abs_diff": round(max(diffs), 3) if diffs else None}  # fmt: skip
    return out


def markdown_table(summary: dict[str, Any], arms: list[str]) -> str:
    rows = ["| arm | n | nDCG@10 | P@5 | MRR known-item (n) | MRR implicit (n) | unjudged@10 | injection in top 3 |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for a in arms:
        c = summary.get(a)
        if not c:
            continue

        def f(key: str, nd: int = 3, c: dict[str, Any] = c) -> str:
            v = c.get(key)
            return "—" if v is None else f"{v:.{nd}f}"

        rows.append(
            f"| {a} | {c['n']} | {f('ndcg10')} | {f('p5')} | {f('mrr')} ({c.get('n_mrr', 0)}) | "
            f"{f('mrr_implicit')} ({c.get('n_mrr_implicit', 0)}) | {f('unjudged10')} | {f('inject_top3', 2)} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
