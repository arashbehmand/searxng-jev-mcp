# Decision providers

The server speaks TypeSafe's System One wire format: `POST {base}/v1/systemone` with a `state` and
typed `questions`, returning typed answers with probabilities. Any server that implements it works.

| `DECISION_PROVIDER` | Base URL | Default model | Context | Rerank batch | K | Notes |
|---|---|---|---|---|---|---|
| `openrouter` | `https://openrouter.ai/api` | `typesafe/jev-1.13` | 32K | 30 | 60 | one key for several decision models; `usage.cost` in every response |
| `typesafe` | `https://api.typesafe.ai` | `jev-1.13.0` | 64K (state ≤ 32K) | 60 | 60 | patch-level pin; no `usage.cost`, so cost is estimated at $0.042/M input tokens |
| `ollaya` | `http://127.0.0.1:11435` | `winnow:e4b` | state ≤ 6,144 tokens | 15 | 30 | local and free; used for `private=true` |
| `custom` | `DECISION_BASE_URL` | `DECISION_MODEL` | — | 30 | 60 | any `/v1/systemone` server; `DECISION_API_KEY` optional |
| `none` | — | — | — | — | — | routing and reranking off (prior ranking) |
| `auto` (default) | | | | | | a runtime chain of every available provider: OpenRouter key → reachable Ollaya → TypeSafe key |

**Failover.** `auto` builds a chain; an explicit provider can add one with `DECISION_FALLBACK`
(e.g. `DECISION_PROVIDER=openrouter` + `DECISION_FALLBACK=typesafe`). When a provider fails fast
(bad key, exhausted spending limit, overload, rate limit, connection error, open circuit) the same call
is retried on the next provider. Timeouts do not fail over: the stage deadline is already spent. A bad
key or an exhausted spending limit (HTTP 401/402/403 "limit") pauses that provider for an hour.

`private=true` never uses a hosted provider: it uses Ollaya if reachable and otherwise no decision
model.

## Behaviour common to all providers

- Stage deadlines come from the provider profile. Hosted: routing 0.8 s, rerank 2 s, assess 1.5 s inside
  the 6 s search budget. Ollaya: routing 4 s, rerank 12 s, assess 6 s and 14 s of extra budget (guesses:
  Ollaya on CPU could not be measured). Late batches keep their prior rank.
- One retry on a connection error, and on HTTP 429 only when `Retry-After` ≤ 1 s (a longer one is
  respected: the call fails and the next provider is tried); no retry on 529/overload.
- API keys are only sent over HTTPS (or HTTP to localhost); a key would otherwise be refused.
- Circuit breaker: 3 consecutive failures open it for 60 s, then a half-open probe.
- An oversized batch (`400 max_tokens_exceeded`, Ollaya `422 STATE_TRUNCATED`) is split in two and
  retried once.
- Error bodies are normalised from OpenRouter `{"error": {"message", "code"}}`, TypeSafe
  `{"detail": {"error_type", "message"}}`, Ollaya `{error, code, detail[]}` and Vercel
  `{message, error_type}`.
- Thresholds (`route_include` 0.35, top-category fallback 0.2, general drop 0.15, engine veto 0.3,
  `tau_keep` 0.2, navigational 0.6, assess stop 0.7 / refine 0.6 / broaden 0.35) are kept per
  provider+model in `decide/providers.py`; the pipeline reads `keep` and `navigational` from the active
  provider. Only the starting defaults are stored today, apart from the calibrated `keep` cut for d1
  and Mercury (below).
- Decision routing (stage B) is off by default (`DECISION_ROUTING=1` enables it): it showed no
  end-to-end gain in the eval. The rerank is on.

## Other decision models on OpenRouter

Two other decision models work on the same endpoint and have tuned profiles. Select them with:

```bash
DECISION_PROVIDER=openrouter
DECISION_MODEL=liquid/d1                        # or inception/mercury-decide:free
```

Measured in a fair comparison on 2026-10-09 (same candidates, prompt and ordering code; three seeded runs;
details in [evaluation.md](evaluation.md#decision-model-comparison-2026-10-09)):

| Model | nDCG@10 (100 queries) | Known-item MRR | Search p95 | USD / 1,000 searches | Profile (batch, deadline, high / medium / keep) |
|---|---|---|---|---|---|
| `typesafe/jev-1.13` (default) | 0.872 | 0.811 | 0.53 s | 0.45 | 30, 2.0 s, 0.80 / 0.40 / 0.20 |
| `liquid/d1` | 0.864 | 0.728 | 1.21 s | 2.05 | 10, 2.0 s, 0.98 / 0.61 / 0.28 |
| `inception/mercury-decide:free` | 0.846 | 0.679 | 1.67 s | 0 | 10, 2.5 s, 0.99 / 0.07 / none |

- d1 and Mercury bill each question against the whole request, so cost grows with batch size squared;
  batch 30 gave no significant gain (d1 +0.008, Mercury +0.002) and costs ~3× more for d1.
- The thresholds differ because the models use the score scale differently (d1 pushes scores to 0 or 1;
  Mercury gives many relevant pages scores near 0). They are set so that "high", "medium" and the
  guard mean the same thing as with Jev.
- `inception/mercury-decide` (without `:free`) has no endpoint. Free models may be rate-limited by
  OpenRouter; none of 2,698 calls in the comparison were.

## Cost

A balanced search with Jev on OpenRouter sends about 7-13K input tokens (routing ~1.5K, rerank
~200 per candidate), about $0.0003-0.0006 per search, or roughly $2-4 a month at 200 searches a day.
Per-call cost is logged to `~/.cache/searxng-jev-mcp/spend.jsonl`; `explain=true` shows it per search.
