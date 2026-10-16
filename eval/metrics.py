"""Ranking metrics, paired bootstrap CIs and weighted kappa. Pure functions, no I/O."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def dcg(gains: Sequence[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at(
    ranked: Sequence[str], grades: dict[str, int], k: int = 10, ideal_pool: Sequence[str] | None = None
) -> float:
    """nDCG@k with graded (0/1/2) labels. Unjudged docs count as 0. The ideal ranking uses every judged
    doc for the query (pass `ideal_pool` to restrict it). Returns 0.0 when nothing is relevant."""
    gains = [grades.get(u, 0) for u in ranked[:k]]
    pool = ideal_pool if ideal_pool is not None else list(grades)
    ideal = sorted((grades.get(u, 0) for u in pool), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at(ranked: Sequence[str], grades: dict[str, int], k: int = 5, min_grade: int = 1) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for u in top if grades.get(u, 0) >= min_grade) / k


def reciprocal_rank(ranked: Sequence[str], targets: set[str]) -> float:
    for i, u in enumerate(ranked, start=1):
        if u in targets:
            return 1.0 / i
    return 0.0


def unjudged_at(ranked: Sequence[str], grades: dict[str, int], k: int = 10) -> float:
    top = ranked[:k]
    return sum(1 for u in top if u not in grades) / len(top) if top else 0.0


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, n: int = 10_000, seed: int = 7, alpha: float = 0.05
) -> dict[str, float]:
    """CI for mean(b − a) over paired per-query values, plus the fraction of resamples with diff ≤ 0."""
    assert len(a) == len(b)
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    if not diffs:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_le0": float("nan")}
    rng = random.Random(seed)
    m = len(diffs)
    stats = []
    for _ in range(n):
        stats.append(sum(diffs[rng.randrange(m)] for _ in range(m)) / m)
    stats.sort()
    lo = stats[int(alpha / 2 * n)]
    hi = stats[min(n - 1, int((1 - alpha / 2) * n))]
    return {"diff": mean(diffs), "lo": lo, "hi": hi, "p_le0": sum(1 for s in stats if s <= 0) / n}


def bootstrap_ci(
    xs: Sequence[float], *, n: int = 10_000, seed: int = 7, alpha: float = 0.05
) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(xs)
    if not m:
        return (float("nan"), float("nan"))
    stats = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return stats[int(alpha / 2 * n)], stats[min(n - 1, int((1 - alpha / 2) * n))]


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], *, labels: Sequence[int] = (0, 1, 2), quadratic: bool = True
) -> float:
    """Cohen's weighted kappa (quadratic by default) between two raters."""
    assert len(a) == len(b) and a
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[idx[x]][idx[y]] += 1
    n = len(a)
    ra = [sum(row) for row in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1)
        return d * d if quadratic else d

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    return 1.0 - num / den if den else 1.0


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else float("nan")


def percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


MetricFn = Callable[[Sequence[str], dict[str, int]], float]
