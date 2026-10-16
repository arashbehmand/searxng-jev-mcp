# Architecture

```
query
  ├─► [A] parse operators, escape SearXNG syntax, scrub        query.py
  │
  ├─► [B] decision routing (one flat request, 0.8 s deadline)  decide/backend.py · route
  │        in parallel with
  └─► [C] general request to SearXNG at t=0                    searxng.py
                                                         │
      [D] vertical request (routed engines, keyword form)       pipeline.py
                                                         │
      [E] merge + dedupe (identifier keys, canonical URLs)      merge.py
                                                         │
      [F] prior = weighted RRF(SearXNG score ranks, BM25F)       lexical.py
                                                         │
      [G] decision rerank (listwise, batches in parallel)       decide/backend.py · rerank
                                                         │
      [H] optional assess: stop / refine / broaden signal       decide/backend.py · assess
                                                         │
      diversity cap → compact text + structuredContent          render.py · server.py
```

`mode=fast` skips B, G and H (static routing, prior ranking). Every decision stage fails open: a
timeout, an error, an open circuit breaker or a missing key falls back to static routing and prior
ranking, with a warning in the output.

## Modules

| Module | Role |
|---|---|
| `server.py` | MCP server (stdio): the `search` tool, input/output schema, debug tools behind `DEBUG_TOOLS=1` |
| `pipeline.py` | Stages, deadlines (6 s balanced / 4 s fast), speculation policy, degradation, paging from the cached pool |
| `query.py` | Operators (`"phrase"`, `site:`, `-site:`, `-term`, `filetype:`), SearXNG-syntax escaping, scrubbing, script/language detection, time-range cues, keyword form for vertical APIs |
| `searxng.py` | HTTP client: `/config` catalog, `/search` with `timeout_limit`, concurrency cap, error classification |
| `route.py` + `catalog.yaml` | Computed general baseline, static keyword routing, per-category engine lists, quota buckets |
| `health.py` | Per-engine EWMA success/latency from `unresponsive_engines`, circuit breaker (2 failures → 15 min; CAPTCHA → 24 h), half-open probes |
| `merge.py` | Canonical URLs, identifier keys (arXiv, DOI, PMID/PMC, Stack Exchange, YouTube, GitHub repo), post-filters, diversity cap |
| `lexical.py` | Tokenizer, BM25F (title ×2, snippet ×1, URL ×0.5; k1 1.2, b 0.3), blended IDF (pool Lucene IDF + wordfreq background), RRF prior, reserved K slots |
| `decide/client.py` | `SystemOneClient`: `POST {base}/v1/systemone`, error-shape normalisation, one retry on 429/connection errors, circuit breaker, rate limiter, spend log |
| `decide/providers.py` | Profiles: OpenRouter, TypeSafe direct, Ollaya, custom; batch sizes, K, timeouts, thresholds per provider+model |
| `decide/questions.py` | Typed questions for route, rerank and assess; web text only inside `state` |
| `decide/backend.py` | `SystemOneDecider`: selection rule, batched rerank with deadline and split-on-oversize, assess verdict mapping |
| `store.py` | sqlite cache (SearXNG responses, decision scores, ranked pools, quota counters) and the JSONL trace |
| `sanitize.py` | HTML/control/zero-width/bidi stripping, prompt-injection heuristic |

## Key rules

- **Engines.** Only engines that are enabled in SearXNG's `/config` are ever named (SearXNG queries
  disabled engines when named). `engines=` is sent, never together with `categories=`.
- **Precedence.** Caller `engines` > caller `categories` > decision routing (when `DECISION_ROUTING=1`)
  or the static keyword map > general baseline. With decision routing on, the model decides the
  verticals and the static map is only the fail-open fallback; engines without their own routing
  question (e.g. Arch wiki, Hoogle) must also pass a keyword gate. A caller category whose engines are
  all unusable (disabled, circuit-open, out of quota) falls back to the general web with a warning.
- **Time ranges.** `time_range` comes only from the caller. Engines without time-range support
  (on many instances Bing, Wikipedia, Stack Overflow, GitHub, arXiv) are left out and listed in a
  `time_range_skipped` warning; words like "latest" or a year only add a soft recency boost.
- **Never an opaque failure.** No usable engine at all, or nothing back before the deadline, is a tool
  error with an explanation; so is any unexpected exception. If every general engine's circuit is
  open they are asked anyway as a probe.
- **Speculation.** The general request starts at t=0. Its results are discarded only when routing
  succeeded, says general web is not useful (p < 0.15), and the vertical request returned results.
  Non-English queries never discard general results.
- **Prior.** One RRF list per SearXNG request ranked by SearXNG's `score` (not array order) plus a
  BM25F list (RRF k = 60, BM25F list weight 1.0, tuned on the eval's dev split); at least 3 vertical results and the top 3 exact-token matches are reserved inside the
  K candidates sent to the decision model.
- **Rerank ordering.** Sort by p; within 0.02 of a group's top result the prior decides; results with
  p < τ_keep (0.2) sort last and are marked `low`; late batches keep their prior slot (`unscored`).
  Authority guard (`AUTHORITY_GUARD`, default `pin3_p80`): a result in SearXNG's general top 3 that ≥ 2
  independent general engines agree on (Bing and Yahoo count once) and that the model rates p ≥ 0.8
  keeps at least that rank (`pin3` guards anything the model does not reject, p ≥ 0.2); variants require p ≥ 0.8 or p within 0.1 of the best, or only guard rank 1.
  A moved result is flagged `guarded` and explained in `why`. Cross-site reposts of one article are
  merged (`also_at`). Suggestions are kept only if they share a word with the query and their added words
  occur in the results. Injection-flagged
  results (`suspect`) are never sent to the model, move five places down in every mode, and their snippet
  is withheld.
- **Flags and options.** In command-line queries (`tar -xvf`, `find . -name`, `ls -la`) dash tokens are
  flags and are sent without the dash; elsewhere `-term` (letters, ≥ 3 chars) is an exclusion,
  forwarded to SearXNG and post-filtered.
- **Concurrency.** Up to 4 concurrent SearXNG requests (a search makes 1-2); the vertical request is
  dispatched before query variants so variants cannot starve it.
- **Errors.** `isError` only for hard failures (SearXNG unreachable, JSON disabled, invalid
  parameters with allowed values). Soft failures go to `warnings`, with `degraded: true`. An empty
  result set is never reported as a failure, and "every engine failed" is never shown as "no results".

## Caching

| What | Key | TTL |
|---|---|---|
| SearXNG response | normalised q, engines, language, time_range, page | 6 h; 10 min with `time_range` or news engines; 60 s if any engine failed or nothing came back |
| Decision score per document | provider, model, prompt version, query, doc title/domain/age/snippet | 7 days |
| Routing decision | provider, model, prompt version, routing state, question set | 7 days |
| Ranked pool (for `page=2..`; versioned, stale shapes ignored; diversity applied across the pool) | all parameters except `page`/`explain` | 10 min |
