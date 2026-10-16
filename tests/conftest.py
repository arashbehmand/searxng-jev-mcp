"""Shared fixtures. Tests never read the developer's .env files and never touch the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from searxng_jev_mcp.config import Settings
from searxng_jev_mcp.health import HealthTable
from searxng_jev_mcp.pipeline import Pipeline
from searxng_jev_mcp.searxng import SearxngClient
from searxng_jev_mcp.store import Cache, Store, TraceLog

FIXTURES = Path(__file__).parent / "fixtures"
SEARX_URL = "http://searx.test"

ENV_VARS = [
    "SEARXNG_URL", "SEARXNG_USERNAME", "SEARXNG_PASSWORD", "DECISION_PROVIDER", "DECISION_MODEL",
    "DECISION_BASE_URL", "DECISION_API_KEY", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_AI_API_KEY",
    "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL", "DEBUG_TOOLS", "SEARXNG_JEV_ENV_FILE", "ENGINE_OVERRIDES",
]  # fmt: skip


def load_fixture(*parts: str) -> Any:
    return json.loads((FIXTURES.joinpath(*parts)).read_text())


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SEARXNG_JEV_KEYRING", "0")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        searxng_url=SEARX_URL,
        decision_provider="none",
        data_dir=tmp_path,
        keyring_enabled=False,
    )


Responder = Callable[[dict[str, str]], httpx.Response | dict[str, Any]]


class FakeSearx:
    """httpx transport that serves /config and /search from synthetic fixtures."""

    def __init__(self, responder: Responder | None = None, config: dict[str, Any] | None = None):
        self.responder = responder or default_responder
        self.config = config or load_fixture("searxng", "config.json")
        self.calls: list[dict[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/config":
            return httpx.Response(200, json=self.config)
        if request.url.path == "/search":
            params = dict(request.url.params)
            self.calls.append(params)
            out = self.responder(params)
            return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def engines_called(self) -> list[list[str]]:
        return [c["engines"].split(",") for c in self.calls]


VERTICAL_ENGINES = {
    "stackoverflow",
    "github",
    "mdn",
    "pypi",
    "askubuntu",
    "superuser",
    "arxiv",
    "pubmed",
    "docker hub",
}


def default_responder(params: dict[str, str]) -> dict[str, Any]:
    engines = set(params["engines"].split(","))
    if engines & VERTICAL_ENGINES:
        body = load_fixture("searxng", "vertical_it.json")
    else:
        body = load_fixture("searxng", "general_python.json")
    body["query"] = params["q"]
    return body


@pytest.fixture
def fake_searx() -> FakeSearx:
    return FakeSearx()


@pytest.fixture
def make_pipeline(settings: Settings, tmp_path: Path) -> Iterator[Callable[..., Pipeline]]:
    stores: list[Store] = []

    def factory(fake: FakeSearx, **kwargs: Any) -> Pipeline:
        store = Store(tmp_path / f"state{len(stores)}.sqlite")
        stores.append(store)
        cache = Cache(store)
        searx = SearxngClient(SEARX_URL, transport=fake.transport)
        return Pipeline(searx, HealthTable(store), cache, trace=TraceLog(tmp_path / "trace.jsonl"), **kwargs)

    yield factory
    for s in stores:
        s.close()
