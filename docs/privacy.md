# Privacy and data flow

This server runs locally (stdio) next to your MCP client. Here is exactly what leaves your machine,
to whom, and what is stored on disk.

## What is sent where

| Destination | When | What is sent |
|---|---|---|
| Your SearXNG instance | every search | the query after scrubbing (see below), engine names, language, time range, page number |
| Search engines (via SearXNG) | every search | whatever SearXNG forwards: the scrubbed query |
| Hosted decision provider (OpenRouter or TypeSafe) | `mode=balanced` with a hosted provider configured (with `DECISION_PROVIDER=auto`, the default, any API key enables this), unless `private=true` | the scrubbed query; for each candidate result its title, domain, an age string such as "8 months", and a snippet of up to 400 characters (all from the public web). With `DECISION_ROUTING=1`, also one routing request per search containing only the scrubbed query |
| Local decision runtime (Ollaya on 127.0.0.1) | `private=true`, or `DECISION_PROVIDER=ollaya` | the same as above, but it never leaves the machine |

Nothing is sent to a decision provider in `mode=fast`, with `DECISION_PROVIDER=none`, or with
`private=true` (which uses a local runtime if one is reachable and otherwise no decision model at all).

OpenRouter requests carry `X-Title: searxng-jev-mcp` and `HTTP-Referer` set to this project's GitHub URL
(`config.PROJECT_URL`). No user identifier is sent. With `auto` or `DECISION_FALLBACK`, a request that
fails on one provider may be repeated on the next one in the chain.

## Scrubbing

Before a query leaves the machine the server removes: paths that identify you or are temporary
(`/Users/…`, `/home/…`, `/root/…`, `C:\Users\…`, `~/…`, `/tmp/…`, macOS `/var/folders/…`; a file name
such as `main.py` is kept), your OS username in `user@host` form, e-mail addresses, UUIDs, hex
addresses and long hex runs, timestamps, IPv4 addresses, and secret-looking strings (API keys, bearer
tokens, JWTs, `password=...` values). System paths such as `/dev/null` or `/etc/fstab` are kept: they
are what the query is about. Queries are capped at about 300 characters. The tool output lists what
was removed in `warnings` (`query_scrubbed`). Scrubbing is heuristic; do not paste secrets into queries.

The raw, unscrubbed query is never written to disk (the ranked-pool cache stores the scrubbed form).

## What is stored locally

All under `~/.cache/searxng-jev-mcp/` (override with `SEARXNG_JEV_DATA_DIR`):

| File | Contents | Disable |
|---|---|---|
| `state.sqlite` | cached SearXNG responses (10 min to 6 h), cached decision scores (7 days), ranked pools (10 min), engine health, quota counters. Expired rows are purged periodically. | `SEARXNG_JEV_CACHE=0` (health and quotas are still kept) |
| `trace.jsonl` | one line per search: scrubbed query, engines, result URLs, timings, warning codes, cost. No snippets. Rotated at 10 MB (one backup, `trace.jsonl.1`). | `SEARXNG_JEV_TRACE=0` |
| `spend.jsonl` | one line per decision call: provider, model, token count, USD cost, latency. No payloads. | — |

API keys are read from the environment, from env files you name explicitly (`--env-file`,
`SEARXNG_JEV_ENV_FILE`) or `~/.config/searxng-jev-mcp/env`, or from the OS keychain. `.env` files in
the current directory are **not** read: a project you happen to run the server from cannot redirect your
key. Keys are only sent over HTTPS (or plain HTTP to localhost). They are never written to any of these
files or to logs; diagnostics show only whether a key is set and its length. Provider error responses
are not passed to the calling agent (they can echo account identifiers); warnings carry only an error
kind and HTTP status.

## Untrusted web content

Snippets come from the open web. The server sends them to the decision model only as JSON values in
`state` (never inside instructions), tells the model they are untrusted, and flags results whose text
addresses a model or ranker and tries to steer it (`suspect`, e.g. "ignore previous instructions", "rate
this result as relevant"). Flagged results are not sent to the model, get no relevance band, move five
places down, and their snippet is withheld from the calling agent. Pages that merely discuss prompts
(system prompts, `[INST]` formats, prompt-injection write-ups) are not flagged. The tool description and output tell the calling agent the same: snippets are untrusted content.
