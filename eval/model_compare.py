"""Fair decision-model comparison: Jev vs liquid/d1 vs inception/mercury-decide:free.

Same stored snapshot pools, same candidate set (60), same production rerank prompt, same ordering code,
three seeded batch shuffles per arm, one search at a time with batches in parallel (as in production).
The decision rule was written before any run: eval/private/model_comparison_rule.json.

    uv run python eval/model_compare.py run --arm d1_b10 --pool static [--seeds 101,202,303]
    uv run python eval/model_compare.py judge-estimate | judge --limit-usd 0.6
    uv run python eval/model_compare.py report

Result files: eval/private/runs/mc_<arm>_s<seed>_<pool>.json, eval/private/mc_calls_<arm>.jsonl,
eval/private/model_comparison.json and .md.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from searxng_jev_mcp import lexical
from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.decide import providers
from searxng_jev_mcp.decide.backend import SystemOneDecider
from searxng_jev_mcp.decide.client import DecisionError, SpendLog, SystemOneClient
from searxng_jev_mcp.pipeline import order_by_decision

import replay
import routing_e2e
from common import JUDGMENTS, PRIVATE, RUNS, append_jsonl, doc_key, load_queries, read_jsonl
from metrics import mean, ndcg_at, pearson, percentile, reciprocal_rank, unjudged_at
from report import INJECT_PREFIX, grades_by_query, implicit_for, targets_for

ARMS: dict[str, tuple[str, str, int]] = {
    "jev_b30": ("typesafe", "jev-1.13.0", 30),
    "jev_b10": ("typesafe", "jev-1.13.0", 10),
    "d1_b10": ("openrouter", "liquid/d1", 10),
    "mercury_b10": ("openrouter", "inception/mercury-decide:free", 10),
    "mercury_b30": ("openrouter", "inception/mercury-decide:free", 30),
    "d1_b30": ("openrouter", "liquid/d1", 30),
}
SEEDS = (101, 202, 303)
OPENROUTER_CAP_USD = 2.50  # this round, including the judge
TYPESAFE_CAP_USD = 1.00
PROD_RERANK_DEADLINE = 2.0
KEY_BEFORE = Path(__file__).parent / "private" / "mc_key_before.json"
STRATA = {
    "code": "code/errors",
    "science": "science",
    "news": "news",
    "i18n": "Persian/non-English",
    "nav": "navigational",
}


def run_path(arm: str, seed: int, pool: str) -> Path:
    return RUNS / f"mc_{arm}_s{seed}_{pool}.json"


def calls_path(arm: str) -> Path:
    return PRIVATE / f"mc_calls_{arm}.jsonl"


# --------------------------------------------------------------------------------------------- spend control
def openrouter_usage(key: str) -> dict[str, Any]:
    d = httpx.get(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=20
    ).json()
    data = d.get("data", {})
    return {k: data.get(k) for k in ("limit", "usage", "limit_remaining")}


def spent_this_round(provider: str) -> float:
    total = 0.0
    for arm, (prov, _, _) in ARMS.items():
        if prov == provider:
            total += sum(float(r.get("cost") or 0) for r in read_jsonl(calls_path(arm)))
    if provider == "openrouter":
        total += sum(float(r.get("cost") or 0) for r in read_jsonl(PRIVATE / "mc_judge_spend.jsonl"))
    return total


# --------------------------------------------------------------------------------------------- running
def make(arm: str, seed: int) -> SystemOneDecider:
    provider, model, batch = ARMS[arm]
    s = load_settings(repo_env_files())
    if provider == "typesafe":
        profile, key = providers.typesafe(model, s.typesafe_base_url), s.key_for("typesafe")
    else:
        profile, key = providers.openrouter(model, s.openrouter_base_url), s.key_for("openrouter")
    profile = dataclasses.replace(profile, batch_size=batch, timeout_s=30.0)
    client = SystemOneClient(profile, key, spend=SpendLog(None))
    decider = SystemOneDecider(client, cache=None)
    decider.shuffle_seed = seed
    orig = client.decide

    async def logged(state: Any, questions: dict[str, Any], **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            r = await orig(state, questions, **kw)
        except DecisionError as e:
            append_jsonl(calls_path(arm), [{"seed": seed, "kind": e.kind, "status": e.status, "s": round(time.perf_counter() - t0, 3),
                                            "n_q": len(questions), "cost": 0.0, "ts": time.time()}])  # fmt: skip
            raise
        append_jsonl(calls_path(arm), [{"seed": seed, "kind": "ok", "status": 200, "s": round(time.perf_counter() - t0, 3),
                                        "n_q": len(questions), "cost": float(r.usage.get("cost") or 0.0),
                                        "input_tokens": r.usage.get("input_tokens"), "ts": time.time()}])  # fmt: skip
        return r

    client.decide = logged  # type: ignore[method-assign]
    return decider


async def rerank_with_backoff(decider: SystemOneDecider, pq: Any, cands: list[Any]) -> dict[str, Any]:
    """Production rerank call; on rate limits / open circuit, wait and retry the unscored docs."""
    attempts, waits, t_first = 0, 0.0, None
    info: dict[str, Any] = {}
    for attempt in range(5):
        todo = [d for d in cands if d.p_jev is None and not d.suspect]
        if not todo:
            break
        t0 = time.perf_counter()
        info = await decider.rerank(pq, todo, navigational=False, deadline=60.0)
        if t_first is None:
            t_first = time.perf_counter() - t0  # per-search latency: the first, production-like attempt
        attempts += 1
        err = info.get("error") or ""
        if "quota" in err:
            raise SystemExit("provider spending limit reached")
        if not any(k in err for k in ("rate_limited", "circuit_open", "overloaded")):
            break
        wait = min(120.0, 10.0 * 2**attempt)
        decider.client.breaker.ok()
        waits += wait
        await asyncio.sleep(wait)
    return {
        "latency_s": round(t_first or 0.0, 3),
        "attempts": attempts,
        "backoff_s": waits,
        "error": info.get("error"),
    }


async def run_arm(arm: str, pool: str, seeds: tuple[int, ...], limit: int = 0) -> None:
    provider = ARMS[arm][0]
    decisions = replay.load_run("route_decisions")
    queries = load_queries()
    if limit:
        queries = queries[:limit]
    s = load_settings(repo_env_files())
    for seed in seeds:
        path = run_path(arm, seed, pool)
        data = json.loads(path.read_text()) if path.exists() else {}
        decider = make(arm, seed)
        for n, e in enumerate(queries):
            if e.id in data:
                continue
            cap = OPENROUTER_CAP_USD if provider == "openrouter" else TYPESAFE_CAP_USD
            if spent_this_round(provider) >= cap * 0.98:
                print(f"{arm}: stopping, {provider} cap reached")
                path.write_text(json.dumps(data))
                return
            if provider == "openrouter" and n % 25 == 0:
                key = s.key_for("openrouter") or ""
                u = openrouter_usage(key)
                before = json.loads(KEY_BEFORE.read_text())["usage"]
                if float(u["usage"] or 0) - float(before) >= OPENROUTER_CAP_USD * 0.98:
                    print(f"{arm}: stopping, key usage shows the round cap reached")
                    path.write_text(json.dumps(data))
                    return
            pq, docs = routing_e2e.pool_docs(e, pool, decisions)
            cands = lexical.select_candidates(docs, 60)
            meta = await rerank_with_backoff(decider, pq, cands)
            data[e.id] = {"scores": {doc_key(d): d.p_jev for d in cands if d.p_jev is not None},
                          "n_cands": len(cands), **meta}  # fmt: skip
            if n % 10 == 0:
                path.write_text(json.dumps(data))
        path.write_text(json.dumps(data))
        await decider.aclose()
        print(f"{arm} seed {seed} {pool}: {len(data)} queries")


# --------------------------------------------------------------------------------------------- rankings
def rankings(arm: str, seed: int, pool: str, view: str, high_p: float = 0.8) -> dict[str, list[str]]:
    run = json.loads(run_path(arm, seed, pool).read_text()) if run_path(arm, seed, pool).exists() else {}
    decisions = replay.load_run("route_decisions")
    out: dict[str, list[str]] = {}
    for e in load_queries():
        r = run.get(e.id)
        if r is None:
            continue
        pq, docs = routing_e2e.pool_docs(e, pool, decisions)
        cands = lexical.select_candidates(docs, 60)
        for d in cands:
            p = r["scores"].get(doc_key(d))
            if p is not None and not d.suspect:
                d.p_jev = p
        ids = {d.id for d in cands}
        guard = "off" if view == "model" else "pin3_p80"
        ordered = order_by_decision(
            cands, pq, keep=0.2, guard=guard, general_engines=replay.GENERAL_ENGINES, high_p=high_p
        )
        out[e.id] = replay.finish(ordered + [d for d in docs if d.id not in ids], waive=bool(pq.sites))
    return out


def baseline_rankings(name: str, pool: str) -> dict[str, list[str]]:
    decisions = replay.load_run("route_decisions")
    out: dict[str, list[str]] = {}
    ce = replay.load_run(f"mc_ce_{pool}") if name == "ce" else {}
    for e in load_queries():
        pq, docs = routing_e2e.pool_docs(e, pool, decisions)
        if name == "prior":
            out[e.id] = replay.finish(docs, waive=bool(pq.sites))
        elif name == "searxng":
            gen = sorted(
                (d for d in docs if "general" in d.request_ranks), key=lambda d: d.request_ranks["general"]
            )
            out[e.id] = [doc_key(d) for d in gen][:20]
        elif name == "ce" and e.id in ce:
            out[e.id] = ce[e.id]["ranking"]
    return out


def ce_run(pool: str) -> None:
    """Cross-encoder baseline on this pool (local model, free)."""
    import math as _m

    decisions = replay.load_run("route_decisions")
    data: dict[str, Any] = {}
    for e in load_queries():
        pq, docs = routing_e2e.pool_docs(e, pool, decisions)
        cands = lexical.select_candidates(docs, 60)
        text = pq.lexical_text or pq.text
        raw = (
            replay.ce_model().predict(
                [(text, f"{d.title} — {d.domain} — {d.snippet}") for d in cands], batch_size=16
            )
            if cands
            else []
        )
        sc = {d.id: 1 / (1 + _m.exp(-float(x))) for d, x in zip(cands, raw, strict=True)}
        cands.sort(key=lambda d: (-sc[d.id], d.prior_rank or 0))
        ids = {d.id for d in cands}
        data[e.id] = {
            "ranking": replay.finish(cands + [d for d in docs if d.id not in ids], waive=bool(pq.sites))
        }
    replay.save_run(f"mc_ce_{pool}", data)


# --------------------------------------------------------------------------------------------- judging
def judge_items() -> list[dict[str, Any]]:
    judged = {(j["qid"], j["key"]) for j in read_jsonl(JUDGMENTS) if j.get("grade") is not None}
    decisions = replay.load_run("route_decisions")
    want: dict[tuple[str, str], None] = {}
    for pool in ("static", "exhaustive"):
        for arm in ARMS:
            for seed in SEEDS:
                if not run_path(arm, seed, pool).exists():
                    continue
                for view in ("model", "shipped"):
                    for qid, r in rankings(arm, seed, pool, view).items():
                        for k in r[:10]:
                            if INJECT_PREFIX not in k and (qid, k) not in judged:
                                want[(qid, k)] = None
        for name in ("prior", "searxng", "ce"):
            for qid, r in baseline_rankings(name, pool).items():
                for k in r[:10]:
                    if INJECT_PREFIX not in k and (qid, k) not in judged:
                        want[(qid, k)] = None
    qs = {e.id: e for e in load_queries()}
    items = []
    by_q: dict[str, dict[str, Any]] = {}
    for qid, key in want:
        if qid not in by_q:
            _pq, docs = routing_e2e.pool_docs(qs[qid], "exhaustive", decisions)
            _pq2, docs2 = routing_e2e.pool_docs(qs[qid], "static", decisions)
            by_q[qid] = {doc_key(d): d for d in docs + docs2}
        d = by_q[qid].get(key)
        if d is not None:
            items.append(
                {"qid": qid, "key": key, "q": qs[qid].q, "url": d.url, "title": d.title, "snippet": d.snippet}
            )
    return items


async def judge(limit_usd: float) -> None:
    import judge as J

    items = judge_items()
    random.Random(11).shuffle(items)
    key = load_settings(repo_env_files()).key_for("openrouter")
    assert key
    jd = J.Judge(J.JUDGE_MODEL, key, limit_usd)
    sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient(headers={"User-Agent": J.UA}, follow_redirects=True) as web:

        async def one(it: dict[str, Any]) -> None:
            async with sem:
                page = await J.fetch_page(web, it["url"])
                msg, src = J.judge_input(it["q"], it["url"], page, it["title"], it["snippet"])
                try:
                    g, reason = await jd.grade(msg)
                except RuntimeError:
                    return
                append_jsonl(JUDGMENTS, [{"qid": it["qid"], "key": it["key"], "url": it["url"], "grade": g,
                                          "reason": reason, "source": src, "model": J.JUDGE_MODEL}])  # fmt: skip

        before = jd.cost
        await asyncio.gather(*(one(it) for it in items))
    append_jsonl(
        PRIVATE / "mc_judge_spend.jsonl", [{"ts": time.time(), "calls": jd.calls, "cost": jd.cost - before}]
    )
    print(f"judged {jd.calls} items, ${jd.cost:.4f}")
    await jd.client.aclose()


# --------------------------------------------------------------------------------------------- report
def per_query_metrics(ranks: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    grades = grades_by_query()
    qs = {e.id: e for e in load_queries()}
    out = {}
    for qid, r in ranks.items():
        g = {k: v for k, v in grades.get(qid, {}).items() if INJECT_PREFIX not in k}
        m = {"ndcg10": ndcg_at(r, g, 10), "ndcg10_c": ndcg_at([k for k in r if k in g], g, 10),
             "unjudged10": unjudged_at(r, g, 10)}  # fmt: skip
        tg = targets_for(qs[qid])
        if tg:
            m["mrr"] = reciprocal_rank(r, tg)
        imp = implicit_for(qs[qid])
        if imp:
            m["mrr_implicit"] = reciprocal_rank(r, imp)
        out[qid] = m
    return out


def avg_over_seeds(per_seed: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid in per_seed[0]:
        rows = [p[qid] for p in per_seed if qid in p]
        keys = set().union(*(r.keys() for r in rows))
        out[qid] = {k: mean([r[k] for r in rows if k in r]) for k in keys}
    return out


def boot_ci(xs: list[float], n: int = 10_000) -> tuple[float, float]:
    rng = random.Random(7)
    st = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return round(st[250], 4), round(st[9750], 4)


def paired(
    a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], metric: str, ids: list[str]
) -> dict[str, float]:
    ids = [i for i in ids if i in a and i in b and metric in a[i] and metric in b[i]]
    diffs = [b[i][metric] - a[i][metric] for i in ids]
    lo, hi = boot_ci(diffs)
    return {"diff": round(mean(diffs), 4), "lo": lo, "hi": hi, "n": len(ids)}


def summarize(pq: dict[str, dict[str, float]], ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("ndcg10", "ndcg10_c", "unjudged10", "mrr", "mrr_implicit"):
        xs = [pq[i][metric] for i in ids if i in pq and metric in pq[i]]
        if xs:
            out[metric] = {"mean": round(mean(xs), 4), "ci": boot_ci(xs), "n": len(xs)}
    return out


def calls_stats(arm: str) -> dict[str, Any]:
    rows = read_jsonl(calls_path(arm))
    if not rows:
        return {}
    lat = [r["s"] for r in rows if r["kind"] == "ok"]
    n = len(rows)
    return {
        "calls": n,
        "call_p50_s": round(percentile(lat, 0.5), 3) if lat else None,
        "call_p95_s": round(percentile(lat, 0.95), 3) if lat else None,
        "error_rate": round(sum(1 for r in rows if r["kind"] != "ok") / n, 4),
        "timeouts": sum(1 for r in rows if r["kind"] == "timeout"),
        "rate_limited": sum(1 for r in rows if r["kind"] in ("rate_limited", "circuit_open")),
        "other_errors": sum(
            1 for r in rows if r["kind"] not in ("ok", "timeout", "rate_limited", "circuit_open")
        ),
        "cost_usd": round(sum(float(r.get("cost") or 0) for r in rows), 5),
    }


def report() -> None:
    queries = load_queries()
    all_ids = [e.id for e in queries]
    splits = {
        "all": all_ids,
        "dev": [e.id for e in queries if e.split == "dev"],
        "test": [e.id for e in queries if e.split == "test"],
    }
    res: dict[str, Any] = {
        "rule": json.loads((PRIVATE / "model_comparison_rule.json").read_text()),
        "pools": {},
    }
    grades = grades_by_query()
    for pool in ("static", "exhaustive"):
        P: dict[str, Any] = {"arms": {}, "baselines": {}}
        avg: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        for arm in ARMS:
            seeds = [s for s in SEEDS if run_path(arm, s, pool).exists()]
            if not seeds:
                continue
            runs = {s: json.loads(run_path(arm, s, pool).read_text()) for s in seeds}
            entry: dict[str, Any] = {"seeds": seeds, "views": {}}
            thr = calibrated_high(arm) if pool == "static" else 0.8
            for view in ("model", "shipped"):
                per_seed = [
                    per_query_metrics(rankings(arm, s, pool, view, high_p=thr if view == "shipped" else 0.8))
                    for s in seeds
                ]
                av = avg_over_seeds(per_seed)
                avg.setdefault(arm, {})[view] = av
                v: dict[str, Any] = {sp: summarize(av, ids) for sp, ids in splits.items()}
                v["seed_ndcg"] = [
                    round(mean([p[i]["ndcg10"] for i in all_ids if i in p]), 4) for p in per_seed
                ]
                v["seed_spread"] = (
                    round(max(v["seed_ndcg"]) - min(v["seed_ndcg"]), 4) if len(per_seed) > 1 else None
                )
                v["strata"] = {lab: round(mean([av[e.id]["ndcg10"] for e in queries if e.stratum == st and e.id in av]), 4)
                               for st, lab in STRATA.items()}  # fmt: skip
                entry["views"][view] = v
            # score stability across seeds and calibration
            rs = []
            for a, b in [(x, y) for i, x in enumerate(seeds) for y in seeds[i + 1 :]]:
                xs, ys = [], []
                for qid, r in runs[a].items():
                    for k, p in r["scores"].items():
                        q2 = runs[b].get(qid, {}).get("scores", {}).get(k)
                        if q2 is not None:
                            xs.append(p)
                            ys.append(q2)
                rs.append(pearson(xs, ys))
            entry["score_r_between_seeds"] = round(mean(rs), 4) if rs else None
            allp = [
                (qid, k, p) for r in runs.values() for qid, x in r.items() for k, p in x["scores"].items()
            ]
            entry["calibration"] = calibration(allp, grades)
            lat = [x["latency_s"] for r in runs.values() for x in r.values() if x.get("latency_s")]
            entry["search_p50_s"] = round(percentile(lat, 0.5), 3) if lat else None
            entry["search_p95_s"] = round(percentile(lat, 0.95), 3) if lat else None
            entry["over_prod_deadline"] = (
                round(sum(1 for x in lat if x > PROD_RERANK_DEADLINE) / len(lat), 4) if lat else None
            )
            entry["coverage"] = round(
                mean([len(x["scores"]) / max(1, x["n_cands"]) for r in runs.values() for x in r.values()]), 4
            )
            P["arms"][arm] = entry
        for name in ("searxng", "prior", "ce"):
            br = baseline_rankings(name, pool)
            if br:
                av = per_query_metrics(br)
                avg.setdefault(name, {})["model"] = av
                avg[name]["shipped"] = av
                P["baselines"][name] = {sp: summarize(av, ids) for sp, ids in splits.items()} | {
                    "strata": {lab: round(mean([av[e.id]["ndcg10"] for e in queries if e.stratum == st and e.id in av]), 4)
                               for st, lab in STRATA.items()}}  # fmt: skip
        P["paired"] = {}
        for view in ("model", "shipped"):
            for a, b in (("jev_b30", "d1_b10"), ("jev_b30", "mercury_b10"), ("d1_b10", "mercury_b10"),
                         ("jev_b30", "jev_b10"), ("mercury_b10", "mercury_b30"), ("d1_b10", "d1_b30")):  # fmt: skip
                if a in avg and b in avg:
                    P["paired"][f"{view}:{a}->{b}"] = {
                        sp: paired(avg[a][view], avg[b][view], "ndcg10", ids) for sp, ids in splits.items()
                    }
        res["pools"][pool] = P
    res["calls"] = {arm: calls_stats(arm) for arm in ARMS}
    for arm, st in res["calls"].items():
        n_searches = sum(
            len(json.loads(run_path(arm, s, p).read_text()))
            for s in SEEDS
            for p in ("static", "exhaustive")
            if run_path(arm, s, p).exists()
        )
        if st and n_searches:
            st["usd_per_1000_searches"] = round(st["cost_usd"] / n_searches * 1000, 3)
    res["decision"] = decide(res)
    (PRIVATE / "model_comparison.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["decision"], indent=1))


def calibration(allp: list[tuple[str, str, float]], grades: dict[str, dict[str, int]]) -> dict[str, Any]:
    ps = [p for _, _, p in allp]
    hist = [0] * 10
    for p in ps:
        hist[min(int(p * 10), 9)] += 1
    judged = [(p, grades[q][k]) for q, k, p in allp if k in grades.get(q, {}) and INJECT_PREFIX not in k]
    bins = {}
    for lo in range(10):
        xs = [g for p, g in judged if min(int(p * 10), 9) == lo]
        if xs:
            bins[f"{lo / 10:.1f}"] = {"n": len(xs), "rel": round(sum(1 for g in xs if g >= 1) / len(xs), 3),
                                      "high": round(sum(1 for g in xs if g == 2) / len(xs), 3)}  # fmt: skip
    return {"n": len(ps), "share_ge_0.8": round(sum(1 for p in ps if p >= 0.8) / len(ps), 4) if ps else None,
            "share_lt_0.2": round(sum(1 for p in ps if p < 0.2) / len(ps), 4) if ps else None,
            "median": round(statistics.median(ps), 3) if ps else None, "histogram_deciles": hist, "by_bin": bins,
            "precision_at": precision_curve(judged)}  # fmt: skip


def precision_curve(judged: list[tuple[float, int]]) -> dict[str, dict[str, float]]:
    out = {}
    for t in [x / 20 for x in range(2, 20)]:
        above = [g for p, g in judged if p >= t]
        if len(above) >= 20:
            out[f"{t:.2f}"] = {"n": len(above), "rel": round(sum(1 for g in above if g >= 1) / len(above), 3),
                               "high": round(sum(1 for g in above if g == 2) / len(above), 3)}  # fmt: skip
    return out


def calibrate() -> dict[str, Any]:
    """Per-model thresholds that give each band the same meaning as Jev's (static pool, all seeds, judged
    documents only):
      high   = lowest t with P(grade 2 | p >= t)  >= Jev's P(grade 2 | p >= 0.8)
      medium = lowest t with P(grade >= 1 | p >= t) >= Jev's P(grade >= 1 | p >= 0.4)
      keep   = highest t with P(grade >= 1 | p < t) <= Jev's P(grade >= 1 | p < 0.2)
    Each side needs at least 30 judged documents."""
    grades = grades_by_query()

    def judged(arm: str) -> list[tuple[float, int]]:
        out = []
        for seed in SEEDS:
            if not run_path(arm, seed, "static").exists():
                continue
            for qid, r in json.loads(run_path(arm, seed, "static").read_text()).items():
                for k, p in r["scores"].items():
                    g = grades.get(qid, {}).get(k)
                    if g is not None and INJECT_PREFIX not in k:
                        out.append((p, g))
        return out

    def above(js: list[tuple[float, int]], t: float, rel: int) -> tuple[float, int]:
        xs = [g for p, g in js if p >= t]
        return (sum(1 for g in xs if g >= rel) / len(xs) if xs else 0.0), len(xs)

    def below(js: list[tuple[float, int]], t: float) -> tuple[float, int]:
        xs = [g for p, g in js if p < t]
        return (sum(1 for g in xs if g >= 1) / len(xs) if xs else 0.0), len(xs)

    ref = judged("jev_b30")
    target_high = above(ref, 0.8, 2)[0]
    target_med = above(ref, 0.4, 1)[0]
    target_keep = below(ref, 0.2)[0]
    grid = [x / 100 for x in range(1, 100)]
    out: dict[str, Any] = {"targets": {"high_p_grade2": round(target_high, 3), "medium_p_rel": round(target_med, 3),
                                       "below_keep_p_rel": round(target_keep, 3)}}  # fmt: skip
    for arm in ARMS:
        js = judged(arm)
        if not js:
            continue
        high = next((t for t in grid if above(js, t, 2)[1] >= 30 and above(js, t, 2)[0] >= target_high), None)
        med = next((t for t in grid if above(js, t, 1)[1] >= 30 and above(js, t, 1)[0] >= target_med), None)
        keep = max(
            (t for t in grid if below(js, t)[1] >= 30 and below(js, t)[0] <= target_keep), default=None
        )
        out[arm] = {"high": high, "medium": med, "keep": keep, "n_judged": len(js),
                    "share_ge_high": round(sum(1 for p, _ in js if high is not None and p >= high) / len(js), 3)}  # fmt: skip
    (PRIVATE / "mc_calibration.json").write_text(json.dumps(out, indent=1))
    return out


def calibrated_high(arm: str) -> float:
    """The 'high' cut for a model: the lowest threshold whose share of grade-2 documents above it matches
    Jev's share above 0.8 (from the stored calibration of the jev_b30 arm). Used for the guard and rel bands."""
    path = PRIVATE / "mc_calibration.json"
    if not path.exists():
        return 0.8
    cal = json.loads(path.read_text())
    return float(cal.get(arm, {}).get("high") or 0.8)


def decide(res: dict[str, Any]) -> dict[str, Any]:
    P = res["pools"]["static"]
    prod = {"jev_b30": "jev", "d1_b10": "d1", "mercury_b10": "mercury"}
    rows = {}
    for arm in prod:
        if arm not in P["arms"]:
            continue
        v = P["arms"][arm]["views"]["shipped"]["all"]
        rows[arm] = {"ndcg10": v["ndcg10"]["mean"], "mrr": v.get("mrr", {}).get("mean"),
                     "p95": P["arms"][arm]["search_p95_s"], "error_rate": res["calls"].get(arm, {}).get("error_rate"),
                     "usd_per_1000": res["calls"].get(arm, {}).get("usd_per_1000_searches")}  # fmt: skip
    if not rows:
        return {}
    best = max(rows, key=lambda a: rows[a]["ndcg10"])
    out = {"best": best, "rows": rows, "checks": {}}
    cands = []
    for arm, r in rows.items():
        key = f"shipped:{best}->{arm}" if arm != best else None
        if key and key not in P["paired"]:
            key2 = f"shipped:{arm}->{best}"
            pr = P["paired"].get(key2, {}).get("all")
            ci = (-pr["hi"], -pr["lo"]) if pr else None
        else:
            pr = P["paired"].get(key, {}).get("all") if key else None
            ci = (pr["lo"], pr["hi"]) if pr else (0.0, 0.0)
        checks = {
            "within_0.01": r["ndcg10"] >= rows[best]["ndcg10"] - 0.01,
            "ci_includes_0": ci is not None and ci[0] <= 0 <= ci[1],
            "mrr_within_0.05": r["mrr"] is not None
            and rows[best]["mrr"] is not None
            and r["mrr"] >= rows[best]["mrr"] - 0.05,
            "p95_ok": r["p95"] is not None and r["p95"] <= PROD_RERANK_DEADLINE,
            "errors_ok": r["error_rate"] is not None and r["error_rate"] < 0.02,
        }
        out["checks"][arm] = checks | {"ci_vs_best": ci}
        if all(checks.values()):
            cands.append(arm)
    out["candidates"] = cands
    out["pick"] = (
        min(cands, key=lambda a: (rows[a]["usd_per_1000"] or 0, -rows[a]["ndcg10"])) if cands else None
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ce", "judge-estimate", "judge", "calibrate", "report", "key"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--pool", default="static")
    ap.add_argument("--seeds", default="101,202,303")
    ap.add_argument("--limit-usd", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (pilot)")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run_arm(a.arm, a.pool, tuple(int(x) for x in a.seeds.split(",")), a.limit))
    elif a.cmd == "ce":
        ce_run(a.pool)
    elif a.cmd == "judge-estimate":
        items = judge_items()
        print(f"unjudged top-10 documents: {len(items)}; estimated judge cost ${len(items) * 0.00058:.3f}")
    elif a.cmd == "judge":
        asyncio.run(judge(a.limit_usd))
    elif a.cmd == "calibrate":
        print(json.dumps(calibrate(), indent=1))
    elif a.cmd == "report":
        report()
    elif a.cmd == "key":
        key = load_settings(repo_env_files()).key_for("openrouter") or ""
        print(json.dumps(openrouter_usage(key)))


if __name__ == "__main__":
    main()
