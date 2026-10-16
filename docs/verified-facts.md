# Verified facts

Behaviour of SearXNG 2026.10.7 and the decision providers that the code relies on, checked live on
2026-10-08. `scripts/check_live.py` re-runs the checks (about 5 live searches and about $0.01 of
decision calls) and writes `eval/private/live_facts.json`. Private instance details (address, engine
health specifics) are not recorded here.

## SearXNG (instance version 2026.10.7)

| Question | Result |
|---|---|
| Is the JSON format enabled? | Yes (200 on `format=json`). |
| Do disabled engines answer when named in `engines=`? | Yes (`wikibooks`, disabled, returned results). The catalog filters to `enabled: true`. |
| Does `positions[i]` align with `engines[i]`? | **Not guaranteed.** Upstream `searx/results.py` keeps `engines` as a `set` and appends `positions` in merge order. The JSON list order of `engines` is therefore set-iteration order, so a result's positions can't be matched to its engines. |
| Does SearXNG parse every word of `q`? | Yes. `pandas <2 :fr !wp` reached engines as `pandas`; `!wp` overrode `engines=yahoo`. |
| Which escaping works? | Inserting a space after the sigil: `pandas < 2 : fr ! wp` reached engines unchanged and `engines=` was respected. A lone `<`, `:` or `!` is never consumed (the parsers need a value after the sigil). |
| Does `!!x` redirect? | Yes: `!!w test` returns HTTP 302 to the external site, even with `format=json`. |
| Does `timeout_limit` return partial results? | Yes: `timeout_limit=0.4` returned in 0.42-0.54 s with every slow engine in `unresponsive_engines` as `[name, "timeout"]`. With all engines timed out the response is HTTP 200 with zero results. |
| Which keys does the JSON response have? | `query, results, answers, corrections, infoboxes, suggestions, unresponsive_engines` (no `number_of_results`). `query` echoes the query after SearXNG's own parsing, which is how escaping was verified. |
| Does the score formula match? | Matches `Π(weights) · n_positions · Σ 1/position` on observed rows (e.g. positions `[7, 2]` → 1.2857). |

## Decision providers via OpenRouter `POST /api/v1/systemone`

| Model id requested | Resolved model | Works | Notes |
|---|---|---|---|
| `typesafe/jev-1.13` | `typesafe/jev-1.13-20260917` | yes | ~196 input tokens per rerank doc (linear). Context 32,000. |
| `liquid/d1` | `liquid/d1-20260930` | yes | Worked without a user id. Quadratic token growth. |
| `inception/mercury-decide` | — | **404** "No endpoints found" | Only `inception/mercury-decide:free` exists (cost 0). Quadratic token growth. |

- Response: `{model, answers: {id: {type: "noul", noul: p}}, usage: {input_tokens, output_tokens, cost}, id, provider}`.
  `usage.cost` is USD. No rate-limit headers.
- Errors (OpenRouter shape `{"error": {"message", "code"}}`):
  bad key → 401 "Missing Authentication header"; unknown question type → 400 with a zod error list
  in `message`; oversized Jev request (~60K tokens) → 400 whose `message` embeds the upstream body
  `{"detail":{"error_type":"max_tokens_exceeded"}}`. The client detects `max_tokens_exceeded` by substring.
- Some 404 error bodies echo an OpenRouter `user_id`; recorders strip `user_id`/`user` keys.
- Burst: 16 concurrent small Jev requests all returned 200 in 0.73 s wall. No limit was hit.
- Decision models are listed by `GET /api/v1/models?output_modalities=decisions` (15 models on
  2026-10-08); the default listing omits them.

Latency, median of 3 / max (seconds), synthetic rerank payloads:

| model | ~2K tokens | ~6K tokens | ~12K tokens |
|---|---|---|---|
| typesafe/jev-1.13 | 0.31 / 0.66 (9 docs) | 0.68 / 0.74 (27 docs) | 0.36 / 0.70 (54 docs) |
| liquid/d1 | 0.36 / 0.39 (2 docs) | 0.43 / 0.52 (5 docs) | 0.52 / 0.52 (9 docs) |
| inception/mercury-decide:free | 0.39 / 0.69 (2 docs) | 0.49 / 0.99 (5 docs) | 0.80 / 1.13 (10 docs) |
