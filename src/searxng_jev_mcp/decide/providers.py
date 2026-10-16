"""Provider profiles for Jev-compatible (`/v1/systemone`) decision backends.

A profile sets the base URL, model, limits and per-model thresholds. Thresholds are stored per
provider+model because calibration differs between models (and possibly between providers serving
the same model). Untuned models get default thresholds; tuning results are in
docs/evaluation.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..config import PROJECT_URL


@dataclass(frozen=True, slots=True)
class Thresholds:
    route_include: float = 0.35  # include a category at p >= this
    route_top: float = 0.2  # ...or the top non-general category at p >= this
    general_drop: float = 0.15  # drop speculative general results only below this
    engine_include: float = 0.3  # engine-level noul
    keep: float = 0.2  # tau_keep: below this a result sorts last and is marked low
    assess_stop: float = 0.7
    assess_refine: float = 0.6
    assess_broaden: float = 0.35
    navigational: float = 0.6
    # Score scale: relevance bands shown to the caller and the authority guard's "high" cut. Models
    # are calibrated differently (docs/evaluation.md, model comparison), so these are per model.
    rel_high: float = 0.8
    rel_medium: float = 0.4
    guard_high: float = 0.8


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    name: str  # openrouter | ollaya | typesafe | custom
    base_url: str
    model: str
    context_tokens: int
    state_tokens: int  # cap for state + longest question
    batch_size: int  # rerank docs per request
    candidates_k: int  # K docs sent to rerank
    max_questions: int = 256
    rate_per_s: float = 20.0
    timeout_s: float = 4.0
    needs_user: bool = False
    auth: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    thresholds: Thresholds = field(default_factory=Thresholds)
    # Snippet budget per doc in rerank payloads (chars); local models get less.
    snippet_chars: int = 400
    # USD per million input tokens, used only when the provider returns no `usage.cost` (TypeSafe direct).
    price_per_m_input: float = 0.0
    # Stage deadlines inside a search (route, rerank, assess) and extra total budget, in seconds.
    # Hosted providers fit the 6 s balanced budget; a local CPU runtime needs much more.
    route_deadline_s: float = 0.8
    rerank_deadline_s: float = 2.0
    assess_deadline_s: float = 1.5
    extra_budget_s: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"


OPENROUTER_HEADERS = {"X-Title": "searxng-jev-mcp", **({"HTTP-Referer": PROJECT_URL} if PROJECT_URL else {})}

# Per provider+model thresholds (defaults unless tuned; see docs/evaluation.md).
THRESHOLDS: dict[str, Thresholds] = {}

# Models whose token accounting re-reads the whole state per question (see docs/verified-facts.md): batch small.
QUADRATIC_MODELS = ("liquid/", "inception/")


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Per-model profile values measured in the model comparison (docs/evaluation.md)."""

    batch_size: int
    timeout_s: float
    rerank_deadline_s: float
    thresholds: Thresholds


# Measured in the model comparison (eval/model_compare.py → eval/private/model_comparison.json and
# mc_calibration.json; docs/evaluation.md). Thresholds give each band the meaning it has for Jev:
#   high/guard: share of top-grade pages above it equals Jev's above 0.8;
#   medium: relevant share above it equals Jev's above 0.4;
#   keep: relevant share below it equals Jev's below 0.2.
# d1 pushes scores to the extremes; Mercury gives many relevant pages scores near 0 (20% of pages under
# 0.01 are relevant), so no "keep" cut is safe for it: 0 means never sort a result last on its score.
MODEL_SETTINGS: dict[str, ModelSettings] = {
    "liquid/d1": ModelSettings(
        batch_size=10,  # billed per question × state: batch 30 costs ~3x more (measured +0.008 nDCG, n.s.)
        timeout_s=4.0,
        rerank_deadline_s=2.0,  # search p95 1.21 s, 1% over 2 s
        thresholds=Thresholds(keep=0.28, rel_high=0.98, rel_medium=0.61, guard_high=0.98),
    ),
    "inception/mercury-decide:free": ModelSettings(
        batch_size=10,  # batch 30 measured the same (+0.002, n.s.)
        timeout_s=4.0,
        rerank_deadline_s=2.5,  # search p95 1.67 s, 3.3% over 2 s
        thresholds=Thresholds(keep=0.0, rel_high=0.99, rel_medium=0.07, guard_high=0.99),
    ),
}


def openrouter(
    model: str = "typesafe/jev-1.13", base_url: str = "https://openrouter.ai/api"
) -> ProviderProfile:
    quadratic = model.startswith(QUADRATIC_MODELS)
    ctx = 32_000 if model.startswith(("typesafe/", "~typesafe/")) else 32_768
    tuned = MODEL_SETTINGS.get(model)
    return ProviderProfile(
        name="openrouter",
        base_url=base_url,
        model=model,
        context_tokens=ctx,
        state_tokens=ctx - 2_000,
        batch_size=tuned.batch_size if tuned else (10 if quadratic else 30),
        candidates_k=60,
        timeout_s=tuned.timeout_s if tuned else 4.0,
        rerank_deadline_s=tuned.rerank_deadline_s if tuned else 2.0,
        extra_headers=dict(OPENROUTER_HEADERS),
        thresholds=tuned.thresholds if tuned else THRESHOLDS.get(f"openrouter:{model}", Thresholds()),
    )


def typesafe(model: str = "jev-1.13.0", base_url: str = "https://api.typesafe.ai") -> ProviderProfile:
    return ProviderProfile(
        name="typesafe",
        base_url=base_url,
        model=model,
        context_tokens=64_000,
        state_tokens=32_000,
        batch_size=60,
        candidates_k=60,
        rate_per_s=40.0,
        price_per_m_input=0.042,
        thresholds=THRESHOLDS.get(f"typesafe:{model}", Thresholds()),
    )


def ollaya(model: str = "winnow:e4b", base_url: str = "http://127.0.0.1:11435") -> ProviderProfile:
    return ProviderProfile(
        name="ollaya",
        base_url=base_url,
        model=model,
        context_tokens=8_192,
        state_tokens=6_144,
        batch_size=15,
        candidates_k=30,
        rate_per_s=4.0,
        timeout_s=20.0,
        auth=False,
        snippet_chars=240,
        # Unmeasured (Ollaya has not been run live yet): generous guesses.
        route_deadline_s=4.0,
        rerank_deadline_s=12.0,
        assess_deadline_s=6.0,
        extra_budget_s=14.0,
        thresholds=THRESHOLDS.get(f"ollaya:{model}", Thresholds()),
    )


def custom(base_url: str, model: str, *, auth: bool) -> ProviderProfile:
    return ProviderProfile(
        name="custom",
        base_url=base_url,
        model=model,
        context_tokens=32_000,
        state_tokens=30_000,
        batch_size=30,
        candidates_k=60,
        auth=auth,
        thresholds=THRESHOLDS.get(f"custom:{model}", Thresholds()),
    )


def with_thresholds(p: ProviderProfile, t: Thresholds) -> ProviderProfile:
    return replace(p, thresholds=t)
