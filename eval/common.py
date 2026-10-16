"""Shared eval helpers: query sets, the frozen dev/test split, snapshot I/O, doc keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from searxng_jev_mcp import merge, query
from searxng_jev_mcp.models import Doc, ParsedQuery
from searxng_jev_mcp.searxng import SearxResponse, parse_response

EVAL = Path(__file__).parent
PRIVATE = EVAL / "private"
SNAPSHOTS = PRIVATE / "snapshots"
RUNS = PRIVATE / "runs"
JUDGMENTS = PRIVATE / "judgments.jsonl"
PAGES = PRIVATE / "pages"


@dataclass(slots=True)
class EvalQuery:
    id: str
    q: str
    stratum: str
    split: str = ""
    hint: str | None = None
    targets: list[str] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)  # implicit labels (private transcripts)
    inject: bool = False
    private: bool = False


def split_of(qid: str, dev_frac: float = 0.6) -> str:
    """Frozen split: a hash of the id, so adding queries never moves existing ones."""
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_frac else "test"


def load_queries(include_private: bool = True) -> list[EvalQuery]:
    out: list[EvalQuery] = []
    pub = yaml.safe_load((EVAL / "queries.public.yaml").read_text(encoding="utf-8"))
    for x in pub["queries"]:
        out.append(EvalQuery(id=x["id"], q=x["q"], stratum=x["stratum"], hint=x.get("hint"),
                             targets=list(x.get("targets") or []), inject=bool(x.get("inject"))))  # fmt: skip
    priv = PRIVATE / "queries.private.yaml"
    if include_private and priv.exists():
        for x in yaml.safe_load(priv.read_text(encoding="utf-8"))["queries"]:
            out.append(EvalQuery(id=x["id"], q=x["q"], stratum=x["stratum"], hint=x.get("hint"),
                                 targets=list(x.get("targets") or []), fetched=list(x.get("fetched") or []),
                                 inject=bool(x.get("inject")), private=True))  # fmt: skip
    for e in out:
        e.split = split_of(e.id)
    return out


def doc_key(d: Doc) -> str:
    """Judgment key: the merge identity (identifier key or canonical URL)."""
    return d.ident or d.canonical


def url_key(url: str) -> str:
    return merge.identifier_key(url) or merge.canonical_url(url)


def target_match(d: Doc, targets: list[str]) -> bool:
    """Known-item match: the doc's canonical URL (or identifier) starts with a target prefix."""
    if not targets:
        return False
    canon = d.canonical
    for t in targets:
        tk = merge.identifier_key("https://" + t)
        if tk and d.ident == tk:
            return True
        tc = merge.canonical_url("https://" + t).rstrip("/")
        if canon.startswith(tc):
            return True
    return False


def snapshot_path(qid: str) -> Path:
    return SNAPSHOTS / f"{qid}.json"


def load_snapshot(qid: str) -> dict[str, Any] | None:
    p = snapshot_path(qid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def snapshot_responses(snap: dict[str, Any]) -> dict[str, SearxResponse]:
    out = {}
    for label, req in snap["requests"].items():
        if req.get("body") is None:
            continue
        out[label] = parse_response(
            req["body"], label=label, engines_asked=req["engines"], elapsed=req["elapsed"]
        )
    return out


def parsed(e: EvalQuery) -> ParsedQuery:
    return query.parse_query(e.q)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
