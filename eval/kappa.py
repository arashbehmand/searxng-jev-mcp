"""Judge agreement check: weighted κ between the LLM judge and reference labels on ~150 pairs.

uv run python eval/kappa.py --sample 150      # writes eval/private/kappa_sample.jsonl (no grades)
# reference labeller writes eval/private/kappa_labels.jsonl: {"id": ..., "grade": 0|1|2}
uv run python eval/kappa.py --score
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from common import JUDGMENTS, PRIVATE, load_queries, read_jsonl
from judge import judge_input, page_path
from metrics import weighted_kappa
from replay import query_pool

SAMPLE = PRIVATE / "kappa_sample.jsonl"
LABELS = PRIVATE / "kappa_labels.jsonl"


def sample(n: int) -> None:
    qs = {e.id: e for e in load_queries()}
    js = [
        j for j in read_jsonl(JUDGMENTS) if j.get("grade") is not None and "seo-spam.example" not in j["key"]
    ]
    rng = random.Random(150)
    by_grade: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for j in js:
        by_grade[j["grade"]].append(j)
    # stratify so every grade is represented (about a third each when available)
    picked: list[dict] = []
    for g in (0, 1, 2):
        rng.shuffle(by_grade[g])
        picked += by_grade[g][: n // 3]
    rest = [j for g in (0, 1, 2) for j in by_grade[g][n // 3 :]]
    rng.shuffle(rest)
    picked += rest[: n - len(picked)]
    rng.shuffle(picked)
    rows = []
    pools: dict[str, dict] = {}
    for i, j in enumerate(picked):
        if j["qid"] not in pools:
            _pq, docs, _ = query_pool(qs[j["qid"]])
            pools[j["qid"]] = {(d.ident or d.canonical): d for d in docs}
        d = pools[j["qid"]].get(j["key"])
        p = page_path(j["url"])
        page = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"ok": False}
        msg, _src = judge_input(qs[j["qid"]].q, j["url"], page, d.title if d else "", d.snippet if d else "")
        rows.append({"id": f"k{i:03d}", "qid": j["qid"], "key": j["key"], "text": msg})
    SAMPLE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} items to {SAMPLE} (grade mix {Counter(j['grade'] for j in picked)})")


def score() -> None:
    sample_rows = {r["id"]: r for r in read_jsonl(SAMPLE)}
    labels = {r["id"]: int(r["grade"]) for r in read_jsonl(LABELS)}
    judged = {(j["qid"], j["key"]): j["grade"] for j in read_jsonl(JUDGMENTS) if j.get("grade") is not None}
    a, b = [], []
    for sid, lab in labels.items():
        r = sample_rows.get(sid)
        if r and (r["qid"], r["key"]) in judged:
            a.append(judged[(r["qid"], r["key"])])
            b.append(lab)
    kq = weighted_kappa(a, b)
    kl = weighted_kappa(a, b, quadratic=False)
    exact = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    conf = Counter(zip(a, b, strict=True))
    out = {"n": len(a), "kappa_quadratic": round(kq, 3), "kappa_linear": round(kl, 3), "exact_agreement": round(exact, 3),
           "confusion_judge_vs_ref": {f"{x}->{y}": c for (x, y), c in sorted(conf.items())}}  # fmt: skip
    (PRIVATE / "kappa_result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.sample:
        sample(a.sample)
    if a.score:
        score()


if __name__ == "__main__":
    main()
