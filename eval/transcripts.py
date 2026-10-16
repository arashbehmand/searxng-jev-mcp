"""Extract past agent web searches from Claude Code transcripts (personal data → eval/private/ only).

For each WebSearch call, records the query and the URLs the agent fetched (WebFetch) within the
next few tool calls of the same session: a secondary, position-biased implicit relevance signal.

    uv run python eval/transcripts.py [--root ~/.claude/projects] [--out eval/private/transcript_queries.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

LOOKAHEAD = 12  # tool calls after a search in which a fetch counts as following from it


def tool_uses(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") if isinstance(rec, dict) else None
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield {
                        "name": block.get("name", ""),
                        "input": block.get("input") or {},
                        "ts": rec.get("timestamp"),
                    }


def extract(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        calls = list(tool_uses(path))
        for i, c in enumerate(calls):
            if c["name"] != "WebSearch":
                continue
            q = c["input"].get("query")
            if not isinstance(q, str) or not q.strip():
                continue
            fetched = []
            for nxt in calls[i + 1 : i + 1 + LOOKAHEAD]:
                if nxt["name"] == "WebSearch":
                    break
                if nxt["name"] == "WebFetch" and isinstance(nxt["input"].get("url"), str):
                    fetched.append(nxt["input"]["url"])
            out.append({"query": q.strip(), "fetched": fetched, "ts": c["ts"], "session": path.stem[:8]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--out", default="eval/private/transcript_queries.jsonl")
    a = ap.parse_args()
    rows = extract(Path(a.root).expanduser())
    seen: set[str] = set()
    uniq = [r for r in rows if not (r["query"].lower() in seen or seen.add(r["query"].lower()))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in uniq), encoding="utf-8")
    print(
        f"{len(rows)} searches, {len(uniq)} unique queries, {sum(1 for r in uniq if r['fetched'])} with fetches → {out}"
    )


if __name__ == "__main__":
    main()
