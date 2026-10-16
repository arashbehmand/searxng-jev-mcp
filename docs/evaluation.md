# Evaluation

Run on 2026-10-08/09 against one self-hosted SearXNG (2026.10.7) with decision models on OpenRouter and
TypeSafe direct. The harness is in `eval/`; snapshots, judgments and personal queries stay in
`eval/private/` (gitignored), so the numbers cannot be reproduced from the public repo alone. The public
query set and every script are included.

## Method

- **Queries:** 100, frozen. 57 public (`eval/queries.public.yaml`: code/errors 15, science 8, news 5,
  how-to 8, navigational 6, operator-heavy 7 incl. SearXNG syntax such as `pandas <2`, Persian 6,
  German 1, Russian 1) and 43 sampled with a fixed seed from the author's own past agent searches.
  22 have known-item targets (MRR without a judge: 16 dev / 6 test); 15 private ones have URLs the agent
  fetched afterwards (implicit labels, position-biased, from a different search engine). 9 public
  queries get an injected spam result. Split 60/40 dev/test by a hash of the id: 58 dev, 42 test.
- **Snapshots:** one exhaustive fan-out per query (2 live searches: all general engines; every enabled
  vertical engine of the routed categories). Every arm is replayed offline on the same frozen pool.
- **Judge:** `google/gemini-3.8-flash` (a model family used nowhere in the pipeline) grading 0/1/2 from
  the fetched page text (78% of documents; 22% from the snippet when the page could not be fetched).
  2,655 judgments ($1.66). Documents that first entered a top 10 after the judging budget ran out
  are unjudged; every table reports unjudged@10, and comparisons
  are also given on **condensed lists** (unjudged documents dropped before scoring).
- **Judge check:** 150 pairs, stratified by judge grade, labelled blind by a second model from a
  different family. Quadratic weighted κ = 0.863 (linear 0.811, exact 84%). This is agreement between
  two model families, **not with humans**; human labels would be the better check.
- **Metrics:** nDCG@10 (graded), P@5 (grade ≥ 1), known-item MRR, implicit MRR, unjudged@10; paired
  bootstrap 95% CIs (10,000 resamples). With 58 dev / 42 test queries, differences below ~0.05
  nDCG@10 are noise.

## Current defaults

`mode=fast` = BM25F + SearXNG prior (RRF k 60, BM25F weight 1.0). `mode=balanced` = the same pool plus
the decision rerank (Jev 1.13, sort by p, near-tie rule, authority guard, suspect demotion). Decision
routing off. Other arms use the same pool and ordering code.

Source: `eval/private/report_current.json` (plain) and `report_current_condensed.json` (condensed),
from `eval/report.py`. Pool: the exhaustive snapshot pool; scores from each arm's stored run; current
ordering (`pin3` guard, repost dedupe, suspect demotion).

**Dev (58 queries)**

| arm | nDCG@10 | condensed | P@5 | known-item MRR (16) | implicit MRR (9) | unjudged@10 |
|---|---|---|---|---|---|---|
| SearXNG order (general request) | 0.643 | 0.646 | 0.693 | 0.922 | 0.302 | 0.009 |
| prior without BM25F | 0.421 | 0.430 | 0.462 | 0.832 | 0.226 | 0.033 |
| **prior (fast mode)** | 0.754 | 0.763 | 0.852 | 0.566 | 0.312 | 0.021 |
| cross-encoder (bge-reranker-v2-m3) | 0.741 | 0.759 | 0.876 | 0.439 | 0.266 | 0.038 |
| **Jev rerank (balanced mode)** | 0.885 | 0.887 | 0.928 | 0.917 | 0.299 | 0.009 |
| Jev blend (α 0.75, no guard) | 0.888 | 0.888 | 0.934 | 0.778 | 0.481 | 0.000 |
| Jev via TypeSafe direct | 0.889 | 0.892 | 0.931 | 0.919 | 0.281 | 0.019 |
| liquid/d1 | 0.879 | 0.885 | 0.931 | 0.914 | 0.310 | 0.015 |
| inception/mercury-decide:free | 0.819 | 0.850 | 0.900 | 0.911 | 0.315 | 0.053 |

**Test (42 queries)**

| arm | nDCG@10 | condensed | P@5 | known-item MRR (6) | implicit MRR (6) | unjudged@10 |
|---|---|---|---|---|---|---|
| SearXNG order | 0.660 | 0.663 | 0.757 | 0.778 | 0.158 | 0.009 |
| prior (fast mode) | 0.761 | 0.779 | 0.890 | 0.441 | 0.206 | 0.036 |
| cross-encoder | 0.738 | 0.764 | 0.871 | 0.472 | 0.227 | 0.050 |
| **Jev rerank (balanced mode)** | 0.855 | 0.865 | 0.952 | 0.783 | 0.155 | 0.029 |
| Jev blend | 0.866 | 0.867 | 0.948 | 0.479 | 0.347 | 0.002 |
| Jev via TypeSafe direct | 0.857 | 0.868 | 0.962 | 0.792 | 0.168 | 0.029 |
| liquid/d1 | 0.856 | 0.871 | 0.952 | 0.792 | 0.147 | 0.038 |

Paired nDCG@10 differences (95% CI), plain / condensed (`pairs_dev`, `pairs_test` in the same files):

| comparison | dev | test |
|---|---|---|
| prior → Jev | +0.131 [+0.094, +0.170] / +0.123 [+0.087, +0.162] | +0.094 [+0.055, +0.138] / +0.086 [+0.047, +0.131] |
| cross-encoder → Jev | +0.144 [+0.098, +0.192] / +0.127 [+0.083, +0.173] | +0.117 [+0.070, +0.163] / +0.101 [+0.057, +0.144] |
| prior → cross-encoder | −0.013 [−0.057, +0.032] / −0.004 [−0.046, +0.039] | −0.023 [−0.069, +0.025] / −0.015 [−0.056, +0.027] |

### Where the decision model does not win, and trade-offs

- **Implicit clicks.** On the 15 queries with URLs the agent actually fetched, Jev does not beat the
  prior (dev 0.30 vs 0.31, test 0.16 vs 0.21). These labels come from another search engine's result
  list, so they are position-biased, but they are the one judge-free signal that does not favour Jev.
- **Known items: the guard copies SearXNG's top slots.** Without the authority guard Jev's known-item MRR
  is 0.74 dev / 0.55 test (production pool, `guard_variants_*.json`) against 0.92 / 0.78 for SearXNG's own
  order; with the `pin3` guard it equals SearXNG's exactly, i.e. the guard restores SearXNG's top
  results rather than improving on them, and it costs nDCG@10 on test (see the guard section).
- **Blend vs sort.** The blend wins on implicit MRR and is level on nDCG, but loses badly on known items
  (it does not use the guard). Sort stays the default.
- **Other decision models.** See the decision model comparison below: d1 is close to Jev on nDCG@10 but
  weaker on known items and 4.6× the cost; Mercury free is 0.026 lower.
- **The cross-encoder** is level with the tuned prior and below it on known items.
- **The judge's rubric mirrors the rerank criteria** ("directly answers the query" ≈ "likely helps
  answer the query"), so an LLM judge may favour an LLM reranker. Snippet leakage check: Jev's AUC for
  "judge grade ≥ 1" is 0.954 on documents the judge graded from the page and 0.981 on those graded from
  the same snippet Jev saw (prior: 0.684 / 0.685). Some leakage, but the page-graded AUC is still high.

### Strata (current defaults, nDCG@10, all 100 queries)

| stratum (n) | SearXNG | prior | Jev |
|---|---|---|---|
| code (32) | 0.659 | 0.771 | 0.863 |
| science (14) | 0.643 | 0.694 | 0.816 |
| news/recent (14) | 0.522 | 0.693 | 0.880 |
| non-English (12) | 0.723 | 0.782 | 0.917 |
| how-to (8) | 0.732 | 0.792 | 0.882 |
| operator-heavy (7) | 0.699 | 0.847 | 0.949 |
| navigational (6) | 0.784 | 0.718 | 0.827 |
| other (7) | 0.503 | 0.806 | 0.888 |

Strata below ~15 queries are anecdotes. Public vs private queries: Jev 0.866 vs 0.880; SearXNG order
0.759 vs 0.506.

### Answering results in the top 5

nDCG@10 is relative to the best possible order of each query's judged pool. A query whose pool holds no
grade-2 page can still score 1.0: if its only partly relevant page sits at rank 1, that is already the
ideal order. 15 of the 100 pools have no grade-2 page at all (23 have two or fewer), and no ranking can
fix those. A plainer measure is how many grade-2 ("directly answers") results land in the top 5 (current
defaults, all 100 queries, from the same per-query runs):

| | SearXNG order | prior (fast mode) | Jev |
|---|---|---|---|
| grade-2 results in top 5 (mean) | 2.35 | 2.61 | 3.05 |
| grade ≥ 1 results in top 5 (mean) | 3.61 | 4.36 | 4.70 |
| queries with at least one grade-2 result in top 5 | 83 | 84 | 84 |
| grade-2 in top 5, queries of 1-7 words (48) | 3.19 | 3.17 | 3.67 |
| grade-2 in top 5, queries of 8+ words (52) | 1.58 | 2.10 | 2.48 |

Jev moves about one more answering result into the top 5. It almost never finds an answer the other
orderings miss entirely (84 vs 83 queries): it reorders what SearXNG returned.

## Routing, end to end

An earlier routing bar used useful-engine recall, which grows with the number of engines asked. The
question that matters is whether the *reranked* results get better. `eval/routing_e2e.py` holds the
decision rerank constant (every candidate of every pool scored with Jev via TypeSafe direct) and varies
the vertical engines.

| pool | vertical engines / search | dev condensed nDCG@10 | vs general-only | test | vs general-only |
|---|---|---|---|---|---|
| general engines only | 0 | 0.893 | — | 0.871 | — |
| + static keyword map (default) | 1.3 | 0.890 | −0.003 [−0.010, +0.002] | 0.872 | +0.001 [0.000, +0.003] |
| + decision routing (model decides) | 2.0 | 0.890 | −0.003 [−0.010, +0.003] | 0.870 | −0.001 [−0.005, +0.003] |
| + decision ∪ static (original rule) | 2.1 | 0.890 | −0.003 [−0.010, +0.003] | 0.870 | −0.001 [−0.005, +0.003] |
| + every vertical engine (16) | 16.1 | 0.895 | +0.002 [−0.008, +0.013] | 0.872 | +0.001 [−0.005, +0.008] |

No pool improves on the general engines alone; the largest stratum difference is −0.017 (navigational,
6 queries).
Known-item MRR is identical across pools. The proxy still says routing finds useful engines (useful-engine
recall vs static +0.268 [+0.054, +0.474] dev, +0.209 [+0.046, +0.381] test), but those engines add
documents the general engines already cover or that the reranker ranks below them.

**Outcome:** decision routing is off by default (`DECISION_ROUTING=1` enables it); balanced mode uses
the static keyword map, whose vertical request starts in parallel with the general one. The original
design had the decision model choose engines; the numbers do not support that yet.

## Decision model comparison (2026-10-09)

Source: `eval/model_compare.py` → `eval/private/model_comparison.json` (all tables below),
`mc_calibration.json`, `runs/mc_<arm>_s<seed>_<pool>.json`, per-call logs `mc_calls_<arm>.jsonl`. Decision
rule written before any run: `eval/private/model_comparison_rule.json`.

Fair set-up: the same 60 candidates per query from the stored snapshots (production pool: general engines
+ static map, repost dedupe), the same production rerank prompt, the same ordering code, three seeded
batch shuffles per arm (d1 batch 30: two, budget), one search at a time with batches in parallel. Jev runs
through TypeSafe direct (same model as on OpenRouter). "As shipped" = `pin3_p80` guard with each model's
own calibrated "high" cut (see calibration); "model only" = guard off. 382 newly surfaced documents were
judged with the same judge; unjudged@10 is 0.5% for every arm.

| arm | nDCG@10 all [CI] | dev | test | known-item MRR | run-to-run spread | search p95 | USD / 1,000 searches |
|---|---|---|---|---|---|---|---|
| **Jev, batch 30** | **0.872** [0.842, 0.897] | 0.871 | 0.872 | 0.811 | 0.004 | 0.53 s | 0.45 |
| Jev, batch 10 | 0.867 [0.837, 0.892] | 0.867 | 0.866 | 0.812 | 0.007 | 0.53 s | 0.47 |
| d1, batch 10 | 0.864 [0.834, 0.890] | 0.867 | 0.860 | 0.728 | 0.008 | 1.21 s | 2.05 |
| d1, batch 30 (2 seeds) | 0.871 [0.842, 0.897] | 0.871 | 0.871 | 0.803 | 0.003 | 1.66 s | 4.64 |
| Mercury free, batch 10 | 0.846 [0.814, 0.874] | 0.842 | 0.851 | 0.679 | 0.010 | 1.67 s | 0 |
| Mercury free, batch 30 | 0.848 [0.816, 0.876] | 0.843 | 0.855 | 0.680 | 0.004 | 1.55 s | 0 |
| prior (fast mode) | 0.778 | 0.775 | 0.782 | 0.504 | — | — | 0 |
| cross-encoder | 0.751 | 0.746 | 0.758 | 0.416 | — | — | 0 (local) |
| SearXNG order | 0.648 | 0.640 | 0.660 | 0.883 | — | — | 0 |

Paired differences in nDCG@10, all 100 queries (as shipped / model only):

| comparison | as shipped | model only |
|---|---|---|
| Jev b30 → d1 b10 | −0.008 [−0.020, +0.004] | −0.010 [−0.023, +0.003] |
| Jev b30 → Mercury b10 | −0.026 [−0.046, −0.008] | −0.029 [−0.050, −0.011] |
| d1 b10 → Mercury b10 | −0.018 [−0.036, −0.003] | −0.019 [−0.036, −0.004] |
| Jev b30 → Jev b10 | −0.005 [−0.014, +0.004] | −0.004 [−0.012, +0.005] |
| Mercury b10 → b30 | +0.002 [−0.006, +0.010] | +0.003 [−0.005, +0.010] |
| d1 b10 → b30 | +0.008 [−0.001, +0.017] | +0.008 [−0.001, +0.016] |

Model only (guard off), all queries: Jev 0.872, d1 b10 0.862, Mercury b10 0.843; known-item MRR 0.650,
0.558, 0.499. Batch size made no significant difference for any model.

Strata (as shipped, nDCG@10; n = 32 / 14 / 14 / 12 / 6):

| | code/errors | science | news | Persian/non-English | navigational |
|---|---|---|---|---|---|
| Jev b30 | 0.863 | 0.830 | 0.862 | 0.911 | 0.833 |
| d1 b10 | 0.860 | 0.801 | 0.856 | 0.927 | 0.820 |
| Mercury b10 | 0.861 | 0.785 | 0.822 | 0.911 | 0.776 |
| prior | 0.789 | 0.721 | 0.695 | 0.867 | 0.736 |

Reliability: 0 errors, 0 timeouts and 0 rate-limit responses in 7,517 calls (Mercury free: 2,698 calls
without a 429). Per-call p95: Jev 0.47 s, d1 1.10 s, Mercury 1.22 s. Searches over the 2 s rerank deadline:
Jev 0.7%, d1 1.0%, Mercury 3.3%. Score stability between seeds (Pearson r): Jev 0.987, d1 0.975,
Mercury 0.938.

Calibration: the models use the score scale very differently. Share of scores ≥ 0.8: Jev 18%, d1 38%,
Mercury 31%; median score 0.36 / 0.56 / 0.03. Thresholds that give each band the meaning it has for Jev
(top-grade share above "high" = Jev's above 0.8; relevant share above "medium" = Jev's above 0.4;
relevant share below "keep" = Jev's below 0.2):

| | high (rel band + guard) | medium | keep (τ_keep) |
|---|---|---|---|
| Jev | 0.80 | 0.40 | 0.20 |
| d1 | 0.98 | 0.61 | 0.28 |
| Mercury free | 0.99 | 0.07 | none: 20% of pages it scores below 0.01 are relevant, so it is set to 0 (never sort a result last on its score) |

These are now the profiles' thresholds (`decide/providers.py`, `MODEL_SETTINGS`).

Secondary pool (exhaustive, one seed): Jev 0.875, d1 b10 0.871, Mercury b10 0.847.

**Decision rule result:** best = Jev batch 30 (0.872). d1 batch 10 is within 0.01 with a CI that includes 0,
but its known-item MRR is 0.083 below Jev's (limit 0.05). Mercury is 0.026 below with a CI that excludes 0.
Only Jev qualifies, so the rule picks **Jev** (it is also the cheapest paid option at $0.45 per 1,000
searches). Jev stays the default. d1 at batch 30 matches Jev on nDCG and
nearly on known items, but costs about 10× more and was not part of the rule (the profile uses batch 10).

## Authority guard variants

Source: `eval/guard_variants.py` → `eval/private/guard_variants_dev.json`, `guard_variants_test.json`,
`guard_variants.md`, `guard_selection.json`. Pool: the production default (general engines + static
vertical map), every candidate scored with Jev 1.13 via TypeSafe direct, repost dedupe on. Mean [95%
bootstrap CI]; known-item MRR has 16 dev / 6 test queries, implicit MRR 9 / 6.

Modes: `off`; `pin3` = a result in SearXNG's general top 3 with ≥ 2 independent engines (Bing/Yahoo count
once) and p ≥ 0.2 is kept at that rank (the previous default); `pin3_p80` = same but p ≥ 0.8; `pin3_band` =
p within 0.1 of the best p; `pin1_*` = SearXNG rank 1 only; `band_reorder` = within the band of results
within 0.1 of the best p, agreed results go first (never lifts a clearly worse result).

**Selection rule, declared before test was computed:** among variants whose dev condensed nDCG@10 is
within 0.005 of the best, take the highest dev known-item MRR; near-ties (< 0.005) → the least override.
Dev chose **`pin3`** (condensed nDCG 0.888, MRR 0.919; eligible: pin3, pin3_band, pin1_band, band_reorder).

| guard | nDCG@10 dev | nDCG@10 test | condensed dev | condensed test | known-item MRR dev | known-item MRR test | implicit MRR dev | implicit MRR test |
|---|---|---|---|---|---|---|---|---|
| off | 0.839 [0.791, 0.880] | 0.842 [0.795, 0.883] | 0.877 [0.831, 0.915] | 0.883 [0.844, 0.917] | 0.738 [0.556, 0.896] | 0.552 [0.218, 0.867] | 0.228 [0.060, 0.448] | 0.263 [0.041, 0.568] |
| pin3 | 0.856 [0.809, 0.894] | 0.841 [0.800, 0.880] | 0.888 [0.842, 0.924] | 0.871 [0.833, 0.906] | 0.919 [0.800, 1.000] | 0.792 [0.542, 1.000] | 0.284 [0.096, 0.514] | 0.167 [0.028, 0.321] |
| pin3_p80 | 0.847 [0.800, 0.886] | 0.843 [0.802, 0.882] | 0.883 [0.837, 0.919] | 0.881 [0.844, 0.914] | 0.875 [0.719, 1.000] | 0.833 [0.500, 1.000] | 0.247 [0.070, 0.471] | 0.179 [0.041, 0.333] |
| pin3_band | 0.850 [0.804, 0.887] | 0.839 [0.793, 0.881] | 0.886 [0.841, 0.921] | 0.879 [0.841, 0.914] | 0.875 [0.719, 1.000] | 0.833 [0.500, 1.000] | 0.284 [0.096, 0.514] | 0.179 [0.041, 0.333] |
| pin1_p80 | 0.843 [0.795, 0.883] | 0.841 [0.795, 0.883] | 0.880 [0.834, 0.916] | 0.881 [0.842, 0.914] | 0.853 [0.674, 1.000] | 0.833 [0.500, 1.000] | 0.228 [0.060, 0.460] | 0.179 [0.041, 0.333] |
| pin1_band | 0.846 [0.800, 0.885] | 0.841 [0.795, 0.883] | 0.884 [0.839, 0.919] | 0.881 [0.842, 0.914] | 0.853 [0.674, 1.000] | 0.833 [0.500, 1.000] | 0.228 [0.060, 0.460] | 0.179 [0.041, 0.333] |
| band_reorder | 0.850 [0.804, 0.887] | 0.839 [0.791, 0.881] | 0.886 [0.841, 0.921] | 0.879 [0.841, 0.913] | 0.865 [0.698, 1.000] | 0.833 [0.500, 1.000] | 0.358 [0.120, 0.627] | 0.179 [0.041, 0.333] |
| _before_pin3_no_dedupe | 0.858 [0.812, 0.895] | 0.843 [0.802, 0.881] | 0.890 [0.845, 0.925] | 0.872 [0.834, 0.907] | 0.919 [0.800, 1.000] | 0.792 [0.542, 1.000] | 0.284 [0.096, 0.514] | 0.167 [0.028, 0.321] |

`_before_pin3_no_dedupe` is the previous default set-up (same guard, no repost dedupe). The exhaustive pool gives
the same ordering of variants (`guard_variants.md`).

Paired differences vs `off` (production pool; `guard_variants_{dev,test}.json`):

| guard | condensed nDCG@10 dev | test | known-item MRR dev | test |
|---|---|---|---|---|
| pin3 | +0.011 [−0.004, +0.026] | −0.012 [−0.034, +0.009] | +0.181 [+0.025, +0.357] | +0.240 [−0.242, +0.661] |
| pin3_p80 (default) | +0.006 [+0.002, +0.011] | −0.002 [−0.012, +0.008] | +0.137 [+0.022, +0.266] | +0.282 [0.000, +0.578] |
| pin3_band | +0.009 [+0.002, +0.019] | −0.003 [−0.012, +0.004] | +0.137 [+0.022, +0.266] | +0.282 [0.000, +0.578] |

**Outcome.** The test run does not confirm the dev choice: `pin3` costs nDCG on test and is behind the
stricter variants on known-item MRR (6 queries, wide CIs). The declared rule selected `pin3`, but the
default was set to `AUTHORITY_GUARD=pin3_p80` because of a design rule fixed before these numbers: the
guard must never pin a result the model rates low above results it rates high. `pin3` violates that by
construction (in the live FastAPI example, the homepage at p 0.36 was lifted to #2). On dev `pin3_p80`
costs 0.005 condensed nDCG@10 (within noise) and 0.044 known-item MRR (16 queries) against `pin3`; the
test numbers were not used to choose it. Every result the guard moves is flagged `guarded` with a reason
in `why`.

An earlier version of this page said the guard left nDCG@10 unchanged; that was wrong. Measured against
the tuned pre-guard ranking, test nDCG@10 moved from 0.874 (`report_tuned.json`) to 0.857 when the guard
was introduced and 0.855 now (`report_current.json`, OpenRouter scores, exhaustive pool, with repost
dedupe); part of that change came from other changes made at the same time (near-tie rule, query
parsing), and the variant table above isolates the guard.

The navigational rerank criterion, on the 7 queries the router flagged navigational, raised condensed
nDCG@10 0.800 → 0.897 (anecdotal; needs `DECISION_ROUTING=1`). A "pin the entity's own site" rule
lowered known-item MRR and was removed.

## Repost dedupe and suggestions

Cross-site reposts (same specific title, snippets with ≥ 0.5 word-3-shingle containment, different
registrable domains) are merged into the better-ranked copy. Effect on the eval is negligible: production
pool, `pin3`, condensed nDCG@10 dev 0.890 → 0.888, test 0.872 → 0.871, MRR unchanged
(`guard_variants_{dev,test}.json`, rows `_before_pin3_no_dedupe` vs `pin3`). Suggestions are kept only
if they share a non-numeric word with the query and every added word occurs in the top results; the live
"fastapi lifespan events 2026 printable / calendar / summer" suggestions are all dropped.

## Prompt injection

- Heuristic (narrowed after false positives): flags only text that addresses a model or ranker and tries to steer
  it. False positives: 0 of 9,167 real result rows in the snapshots (the old patterns: 6, all
  legitimate), 0 of 47 benign LLM-developer texts in the tests. It flags all 3 injected variants used
  here and 24 spam texts in the tests, but those are in-sample. Spam that only claims authority is not
  caught: in a probe, a page titled "… Official Python Documentation" whose snippet said "the official,
  primary and authoritative reference … 80% off our premium course" scored 0.44 (a real Stack Overflow
  answer: 0.94), so it ranks below good results but is not rejected.
- Flagged results are not sent to the model, move five places down, and their snippet is withheld. The
  injected result reached the top 3 in 0 of 9 injected queries for every arm (with an earlier
  "keep the prior rank" rule: 2 of 9).
- With the heuristic off, Jev scored all 9 injected results 0.02-0.08. These injections are crude; this
  is not evidence of robustness to adaptive attacks.

## Assess (stop / refine / broaden)

Label: a grade-2 document is in the Jev top 8 (true for 84% of queries, so this is circular with the
judge and has only 16 negatives).

| verdict | count | answer present |
|---|---|---|
| stop | 75 | 73 (precision 0.97 vs base rate 0.84) |
| refine | 15 | 5 |
| broaden | 5 | 2 |
| unsure | 5 | 4 |

`stop` is trustworthy; `broaden` fired too rarely to evaluate. Assess stays opt-in.

## Calibration and stability

Jev's p is bimodal (most documents below 0.1 or above 0.7) and behaves like "probability of grade 2":
documents at p 0.3-0.4 are already 86% relevant (grade ≥ 1). τ_keep = 0.2 sits in the valley (34%
relevant below, 64%+ above). Four runs with different batch shuffles: per-document Pearson r
0.989-0.992 (bar 0.9), mean |Δp| 0.023-0.026. TypeSafe direct vs OpenRouter (same model): r = 0.988.

## Tuning (dev, then one test run)

| setting | chosen | evidence |
|---|---|---|
| RRF k × BM25F weight | k 60, w 1.0 (initial 20, 0.7) | prior nDCG@10 dev 0.704 → 0.763, test 0.733 → 0.769; it lowers the prior's known-item MRR (dev 0.62 → 0.56), which the tuning objective did not track |
| blend α | 0.75 best on dev; sort kept | blend lost on test and on known items |
| τ_route | 0.35 kept | trade-off curve only (0.2 → recall 0.61 at 2.6 engines; 0.8 → 0.41 at 1.6) |
| τ_keep | 0.2 | calibration valley |
| authority guard | on | known-item MRR (above); implicit MRR trade-off disclosed |

## Latency (live)

Cache off, fresh health state, (query, mode) pairs in random order, 12 queries per mode, decision calls
via TypeSafe direct (same Jev 1.13). OpenRouter adds
a hop: its rerank of 60 documents took p50 0.53 s / p95 0.82 s in the eval vs 0.29 / 0.49 s direct, and
its routing p95 was 0.70 s against the 0.8 s deadline.

| mode | p50 | p95 | max | target p50 / p95 / hard |
|---|---|---|---|---|
| fast | 0.70 s | 2.04 s | 2.05 s | 1.5 / 3 / 4 s |
| balanced (default: rerank, static routing) | 1.04 s | 1.78 s | 2.19 s | 2.3 / 5 / 6 s |
| balanced + decision routing | 1.01 s | 2.21 s | 3.27 s | 2.3 / 5 / 6 s |

An earlier run (n = 10, fixed order) gave assess p50 1.26 s / p95 1.39 s.

## Adoption bar

The bar was written down before any results: a decision stage is on by default if its dev gain is
≥ 0.05 nDCG@10 (≥ 10% useful-engine recall for routing) with a CI excluding 0 and p95 latency within
budget, and it beats the cross-encoder arm.

| stage | against the bar (dev) | end-to-end check | default |
|---|---|---|---|
| Rerank | +0.131 nDCG@10 vs tuned prior [+0.094, +0.170]; beats cross-encoder +0.144; p95 1.78 s: **pass** (`report_current.json`, `latency.json`) | same | **on** |
| Routing | useful-engine recall +0.268 [+0.054, +0.474]; p95 within budget: **pass** | end-to-end nDCG@10 vs general-only −0.003 [−0.010, +0.003]: **fail** | **off** |
| Assess | no bar defined | — | opt-in |

An earlier analysis (initial prior settings, no authority guard) gave the same rerank conclusion:
prior → Jev +0.172 [+0.124, +0.223] dev, +0.146 test; cross-encoder → Jev +0.121 dev.

Caveats for everything above: one SearXNG instance over two days; an LLM judge whose rubric resembles
the rerank question, checked against a second model family rather than people; 58 dev queries; the test
split also informed two decisions (blend kept off; the injection-demotion change), so it is not fully
untouched.
