# Contributing

Thanks for helping. Small, focused pull requests with tests are easiest to review.

## Setup

```bash
uv sync --dev
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

All tests run offline against synthetic fixtures in `tests/fixtures/`. They never read your `.env`
files and never touch the network. CI runs the same commands on Python 3.12 and 3.13.

The server never reads `.env` files from the working directory. The repo's own dev scripts
(`scripts/`, `eval/`) load the repository's `.env` and `.env.local` explicitly
(`config.repo_env_files()`), and `scripts/mcp_client.py` passes them to the server with `--env-file`.

## Rules for this repository

- **No private data.** Never commit real SearXNG hostnames or IP addresses, API keys, personal queries,
  or snapshots of real result pages. Fixtures must be synthetic or scrubbed. Personal eval data
  belongs in `eval/private/` (gitignored). `.env`, `.env.local` and `NOTES.local.md` are gitignored.
- **Logs go to stderr.** stdout carries the MCP protocol.
- **Fail open.** A decision-model failure must degrade to prior ranking with a warning, never to an
  error or an empty result. Only SearXNG being unreachable, JSON being disabled, or invalid parameters
  are tool errors.
- **Untrusted web text** goes into decision-model `state` as JSON values only, never into
  `instructions` or `criteria`.
- **Be gentle with SearXNG.** Keep concurrency at 2 or less in anything that hits a live instance;
  quota-limited engines (Stack Exchange, GitHub, Google Scholar, Google CSE) get banned quickly.

## Evaluation

Ranking or routing changes should come with eval numbers. The harness is in `eval/`:

```bash
uv run python eval/snapshot.py            # live: 2 searches per query, raw JSON → eval/private/snapshots
uv run python eval/replay.py --arms prior,jev
uv run python eval/judge.py --arms prior,jev
uv run python eval/report.py --arms prior,jev --stage-arms jev
```

The public query set is `eval/queries.public.yaml` (public topics only). Snapshots, judgments and
personal queries stay in `eval/private/`. See [docs/evaluation.md](docs/evaluation.md).

## Adding a decision provider

Any server that implements `POST /v1/systemone` (TypeSafe System One wire format) works with
`DECISION_PROVIDER=custom` and `DECISION_BASE_URL`. To add a named profile, extend
`src/searxng_jev_mcp/decide/providers.py` (context limit, batch size, K, timeouts, thresholds) and
`decide/factory.py`, and add tests with a fake transport like those in `tests/test_decide.py`.
