"""Assess eval: does `coverage.verdict` agree with "is a direct answer in the top 8?".

Label: some doc in the production (jev arm) top 8 is graded 2 by the judge. Assess runs live on those
8 docs (title, domain, age, snippet), one small decision call per query.

    uv run python eval/assess_eval.py --run      # live calls → runs/assess.json
    uv run python eval/assess_eval.py            # metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from searxng_jev_mcp.config import load_settings, repo_env_files
from searxng_jev_mcp.decide.backend import verdict
from searxng_jev_mcp.decide.providers import Thresholds

from common import doc_key, load_queries
from replay import load_run, make_decider, query_pool, save_run
from report import grades_by_query


async def run(force: bool) -> None:
    jev = load_run("jev")
    data = load_run("assess")
    decider = make_decider("jev", load_settings(repo_env_files()))
    for e in load_queries():
        if e.id not in jev or (e.id in data and not force):
            continue
        pq, docs, _ = query_pool(e)
        by_key = {doc_key(d): d for d in docs}
        top = [by_key[k] for k in jev[e.id]["ranking"][:8] if k in by_key]
        try:
            a = await decider.assess(pq, top, None, 30.0)
            data[e.id] = {"answered": a.answered, "ambiguous": a.ambiguous, "has_primary": a.has_primary,
                          "top8": [doc_key(d) for d in top]}  # fmt: skip
        except Exception as ex:
            print(e.id, ex)
    save_run("assess", data)
    print(f"assess cost ${decider.client.spend.total_cost:.4f}")
    await decider.aclose()


def metrics(stop: float = 0.7, refine: float = 0.6, broaden: float = 0.35, split: str = "") -> dict:
    grades = grades_by_query()
    data = load_run("assess")
    qs = {e.id: e for e in load_queries()}
    t = Thresholds(assess_stop=stop, assess_refine=refine, assess_broaden=broaden)
    conf: Counter[tuple[str, bool]] = Counter()
    pairs = []
    for qid, a in data.items():
        if split and qs[qid].split != split:
            continue
        g = grades.get(qid, {})
        judged = [k for k in a["top8"] if k in g]
        if len(judged) < 6:
            continue
        label = any(g.get(k, 0) == 2 for k in a["top8"])
        v = verdict(a["answered"], a["ambiguous"], t)
        conf[(v, label)] += 1
        pairs.append((a["answered"], label))
    n_stop = conf[("stop", True)] + conf[("stop", False)]
    n_broaden = conf[("broaden", True)] + conf[("broaden", False)]
    # p(answered) calibration by bin
    bins: dict[str, list[bool]] = {}
    for p, lab in pairs:
        b = f"{min(int(p * 5), 4) / 5:.1f}-{min(int(p * 5), 4) / 5 + 0.2:.1f}"
        bins.setdefault(b, []).append(lab)
    return {
        "n": len(pairs),
        "base_rate_answer_in_top8": round(sum(lab for _, lab in pairs) / len(pairs), 3) if pairs else None,
        "confusion": {f"{v}|{'answer' if lab else 'no_answer'}": c for (v, lab), c in sorted(conf.items())},
        "stop_precision": round(conf[("stop", True)] / n_stop, 3) if n_stop else None,
        "stop_rate": round(n_stop / len(pairs), 3) if pairs else None,
        "broaden_precision": round(conf[("broaden", False)] / n_broaden, 3) if n_broaden else None,
        "broaden_rate": round(n_broaden / len(pairs), 3) if pairs else None,
        "answered_calibration": {
            b: {"n": len(v), "frac_answer": round(sum(v) / len(v), 2)} for b, v in sorted(bins.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.run:
        asyncio.run(run(a.force))
    print(
        json.dumps({"all": metrics(), "dev": metrics(split="dev"), "test": metrics(split="test")}, indent=1)
    )


if __name__ == "__main__":
    main()
