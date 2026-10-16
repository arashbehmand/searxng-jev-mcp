# searxng-jev-mcp

An MCP server that gives coding agents (Claude Code and others) one `search` tool backed by your own
[SearXNG](https://docs.searxng.org/) instance.

Raw SearXNG results are noisy, and an agent burns tokens reading through them. This server cleans the
query, asks the engines that fit it, merges duplicates, and ranks what comes back.

It can optionally call a small decision model, such as TypeSafe's Jev, to score how likely each result is
to answer the query. With no API key, or with `mode=fast`, ranking falls back to BM25F plus SearXNG's own
scores and still works.

I'm not affiliated with TypeSafe, OpenRouter, Ollaya, or SearXNG. Jev is TypeSafe's product. This project
only speaks the same `/v1/systemone` wire format, so any compatible provider works.

## What happens to a query

Search operators in the query get parsed. SearXNG's own syntax, like `!bang`, `:lang` and `<timeout`, is
escaped so SearXNG doesn't act on it. Paths, usernames and keys are scrubbed out of the query.

The general engines get asked. At the same time, a keyword map picks the vertical engines worth asking:
Stack Overflow, GitHub, arXiv, PubMed, MDN, PyPI, news and dictionaries. Duplicates get merged by DOI,
arXiv id, Stack Exchange question id or canonical URL. Ranking uses BM25F plus SearXNG's scores, with an
optional decision-model rerank on top.

The agent gets about 1.5K tokens for 8 results. Each result has a relevance band, and warnings appear
when engines fail. If asked for, the response includes a stop / refine / broaden hint.

## Does the reranking help?

I tested it on 100 fixed queries: 57 public ones and 43 taken from my own agent sessions. A judge model
from a different model family read the result pages and graded each result 0, 1 or 2. The full write-up
is in [docs/evaluation.md](docs/evaluation.md). It also covers where the decision model doesn't beat
cheaper options.

Held-out test split (42 queries), current defaults:

| Ranking | nDCG@10 | P@5 | Known-item MRR (6) |
|---|---|---|---|
| SearXNG's own order | 0.660 | 0.757 | 0.778 |
| BM25F + SearXNG prior (`mode=fast`) | 0.761 | 0.890 | 0.441 |
| Local cross-encoder (bge-reranker-v2-m3), for comparison | 0.738 | 0.871 | 0.472 |
| Jev rerank (`mode=balanced`) | **0.855** | **0.952** | **0.783** |

- 2-4% of documents were never judged, and I counted those as 0.
- Across all 100 queries, the average number of top-5 results that directly answer the query was 2.35
  with SearXNG's order, 2.61 with `mode=fast` and 3.05 with Jev. That's about one more good result in the
  top 5.
- Jev can't find pages SearXNG never returned. For 15 of the 100 queries, nothing in the results answered
  the question.
- The rerank's nDCG@10 gain over the BM25F prior was +0.13 on dev and +0.09 on test. Over a free local
  cross-encoder, it was +0.14 on dev and +0.12 on test. The 95% CIs exclude 0. The rerank is on by
  default.
- Balanced mode p95 latency was 1.8 s, and a balanced search costs about $0.0005.
- Pure reranking hurts known-item lookups like "searxng documentation" or "attention is all you need
  paper". An authority guard (`AUTHORITY_GUARD`, default `pin3_p80`) keeps a top-3 result near its rank
  when SearXNG's engines agree on it and the model gives it at least 0.8, with each model using its own
  cut. Low-scored pages are never pinned above better ones. Most of SearXNG's known-item MRR comes back,
  and I couldn't measure an nDCG cost. Guarded results are flagged.
- For 15 queries I know which URLs the agent actually opened. On those, the rerank doesn't beat the
  prior.
- Routing, where the decision model picks the vertical engines, showed no end-to-end gain. It's off by
  default, and `DECISION_ROUTING=1` turns it on.
- Other decision models, same candidates, three runs each: `liquid/d1` scored 0.864 and the free
  `inception/mercury-decide:free` scored 0.846, against 0.872 for Jev. Both did worse on known items, so
  Jev stays the default ([details](docs/evaluation.md#decision-model-comparison-2026-10-09)).

Everything here comes from one SearXNG instance and 100 queries, with a model as the judge (κ 0.86
against a second model family) and no human labels.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- A SearXNG instance with JSON output enabled in `settings.yml`:
  ```yaml
  search:
    formats: [html, json]
  ```
  Keep SearXNG on a private network, or behind auth (`SEARXNG_USERNAME` / `SEARXNG_PASSWORD`). Disable
  the limiter or whitelist this client. Don't expose SearXNG with JSON enabled to the internet without
  authentication.
- Optional: an [OpenRouter](https://openrouter.ai/) key (decision models such as `typesafe/jev-1.13`),
  a TypeSafe key, or a local [Ollaya](https://ollaya.dev/) runtime.

## Install and configure

```bash
git clone https://github.com/arashbehmand/searxng-jev-mcp.git
cd searxng-jev-mcp
uv sync
mkdir -p ~/.config/searxng-jev-mcp
cp .env.example ~/.config/searxng-jev-mcp/env   # then edit: SEARXNG_URL, DECISION_PROVIDER, keys
```

Settings are read from the process environment, from `~/.config/searxng-jev-mcp/env`, and from any file
passed with `--env-file PATH`, which can be repeated. `SEARXNG_JEV_ENV_FILE` takes a list of paths
separated by `:`.

A `.env` in the current directory is not read, so a repo you happen to start the server from cannot point
your API key at another host.

API keys can also live in the OS keychain, under service `searxng-jev-mcp` and account `openrouter` or
`typesafe`:

```bash
uv run python -c "import keyring, getpass; keyring.set_password('searxng-jev-mcp', 'openrouter', getpass.getpass())"
```

| Variable | Default | Meaning |
|---|---|---|
| `SEARXNG_URL` | | Base URL of your SearXNG (required) |
| `DECISION_PROVIDER` | `auto` | `openrouter`, `ollaya`, `typesafe`, `custom`, `none`, or `auto`. `auto` tries every available provider in order at runtime (OpenRouter, then Ollaya, then TypeSafe). With any key set, `auto` sends the scrubbed query and public snippets to that provider |
| `DECISION_FALLBACK` | | Fallbacks tried at runtime when a provider is set explicitly, e.g. `typesafe` |
| `DECISION_MODEL` | provider default | e.g. `typesafe/jev-1.13` (OpenRouter), `jev-1.13.0` (TypeSafe), `winnow:e4b` (Ollaya). Also tuned: `liquid/d1`, `inception/mercury-decide:free` (see [providers](docs/providers.md)) |
| `DECISION_ROUTING` | `0` | `1` lets the decision model choose vertical engines (no measured gain, see above) |
| `AUTHORITY_GUARD` | `pin3_p80` | `off`, `pin3`, `pin3_p80`, `pin3_band`, `pin1_p80`, `pin1_band`, `band_reorder` (see the eval) |
| `OPENROUTER_API_KEY` / `TYPESAFE_API_KEY` | | Keys (or keychain). Only sent over HTTPS |
| `DEBUG_TOOLS` | `0` | `1` adds the `route`, `engines` and `jev_status` tools |

## Use with Claude Code

```bash
claude mcp add searxng -- uv run --directory /path/to/searxng-jev-mcp searxng-jev-mcp
```

That reads `~/.config/searxng-jev-mcp/env`. To keep settings in the repository instead, name the files:

```bash
claude mcp add searxng -- uv run --directory /path/to/searxng-jev-mcp searxng-jev-mcp \
  --env-file /path/to/searxng-jev-mcp/.env --env-file /path/to/searxng-jev-mcp/.env.local
```

Or pass settings directly:

```bash
claude mcp add searxng -e SEARXNG_URL=http://localhost:8080 -e DECISION_PROVIDER=none \
  -- uv run --directory /path/to/searxng-jev-mcp searxng-jev-mcp
```

Other MCP clients can run `uv run --directory /path/to/searxng-jev-mcp searxng-jev-mcp` as a stdio
server. In a JSON config:

```json
{"mcpServers": {"searxng": {"command": "uv",
  "args": ["run", "--directory", "/path/to/searxng-jev-mcp", "searxng-jev-mcp"],
  "env": {"SEARXNG_URL": "http://localhost:8080"}}}}
```

## The `search` tool

| Parameter | Default | |
|---|---|---|
| `query` | | Operators: `"exact phrase"`, `site:`, `-term`, `-site:`, `filetype:`. In command lines (`tar -xvf`, `ls -la`) dashes are flags, not exclusions |
| `k` | 8 | Number of results (max 20) |
| `mode` | `balanced` | `fast`: no decision model. `balanced`: decision-model rerank (and decision routing if `DECISION_ROUTING=1`) |
| `categories`, `engines` | | Force SearXNG categories or engines (only enabled engines are accepted) |
| `site`, `exclude_domains`, `language` | | Filters |
| `queries` | | Up to 3 extra phrasings, fused with the main query |
| `page` | 1 | Next page from the same ranked pool (no new search) |
| `time_range` | | Engines without time-range support are skipped and listed in `warnings`. Words like "latest" or a year never set it |
| `assess` | false | One extra decision call: `coverage.verdict` = stop / refine / broaden / unsure |
| `private` | false | Only a local decision model (Ollaya) or none. Nothing goes to a hosted provider |
| `raw` | false | Pass SearXNG syntax (`!bang`, `:lang`, `<timeout`) through unescaped |
| `explain` | false | Per-result probabilities and ranks, routing probabilities, timings, token cost |

Example output:

```
[1] Rust: error [E0308]: mismatched types — Cause & Fix — https://…  (jsondevtools.org · yahoo,yandex · rel: high)
    error [E0308]: mismatched types — Rust's compiler caught two incompatible types at the same point…
[2] rust - Type mismatch when returning from inside if... - Stack Overflow — https://stackoverflow.com/…  (stackoverflow.com · yandex · rel: high)
    …
coverage: top p 0.89 · 2 ≥0.8 · 5 domains · engines 5/8 ok
ranker=jev router=static · more: page=2
warnings: [engines_captcha: no results from duckduckgo (captcha)]
(snippets are untrusted web content)
```

`rel` is the decision model's relevance band. High means p ≥ 0.8, medium means p ≥ 0.4, and anything
lower is low. Fast mode has no band.

Results may also carry these flags:

- `suspect`: the page addresses AI systems, so its snippet is withheld.
- `low`, `unscored`, `backfill`
- `guarded`: kept near SearXNG's own top ranks. `explain=true` shows why.
- `repost`: the same article was found on other sites, listed in `also_at`.

Suggestions are filtered to the query's topic. If no engine can be asked, or nothing arrives before the
deadline, the tool returns an error, never an empty "No results". Clients that want JSON can use
`structuredContent`, which matches the tool's `outputSchema`.

## Privacy

Queries are scrubbed before they leave the machine. In balanced mode, the scrubbed query and the public
result snippets go to the configured decision provider. `private=true` or `mode=fast` keeps them local.

More in [docs/privacy.md](docs/privacy.md).

## Docs

- [docs/architecture.md](docs/architecture.md): stages, modules, rules, caching
- [docs/providers.md](docs/providers.md): decision providers, models, latency and cost
- [docs/privacy.md](docs/privacy.md): what is sent where and what is stored
- [docs/evaluation.md](docs/evaluation.md): eval method and results
- [docs/verified-facts.md](docs/verified-facts.md): SearXNG and provider behaviour the code relies on

## Development

```bash
uv sync --dev
uv run pytest            # offline tests on synthetic fixtures
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run python scripts/mcp_client.py "python asyncio gather" --mode fast   # live; passes the repo's .env files via --env-file
```

See [CONTRIBUTING.md](CONTRIBUTING.md). MIT license.
