"""Selecting liquid/d1 or Mercury free via DECISION_PROVIDER=openrouter + DECISION_MODEL, and per-model
thresholds reaching the relevance bands and the guard."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from searxng_jev_mcp import config
from searxng_jev_mcp.decide import providers
from searxng_jev_mcp.decide.backend import SystemOneDecider
from searxng_jev_mcp.decide.client import SystemOneClient
from searxng_jev_mcp.decide.factory import build_deciders
from searxng_jev_mcp.pipeline import Pipeline, SearchParams
from searxng_jev_mcp.render import build_output

from .conftest import FakeSearx
from .test_decide import FakeProvider

MakePipeline = Callable[..., Pipeline]
MODELS = ["liquid/d1", "inception/mercury-decide:free"]


@pytest.mark.parametrize("model", MODELS)
def test_model_selectable_through_openrouter(model: str, tmp_path: Path) -> None:
    s = config.Settings(_env_file=None, searxng_url="http://x", decision_provider="openrouter",  # type: ignore[call-arg]
                        decision_model=model, openrouter_api_key="k", data_dir=tmp_path, keyring_enabled=False)  # fmt: skip
    d, _private, reason = build_deciders(s, None)
    assert reason is None and isinstance(d, SystemOneDecider)
    tuned = providers.MODEL_SETTINGS[model]
    p = d.profile
    assert p.model == model and p.base_url.startswith("https://openrouter.ai")
    assert p.batch_size == tuned.batch_size and p.timeout_s == tuned.timeout_s
    assert p.rerank_deadline_s == tuned.rerank_deadline_s and p.thresholds == tuned.thresholds
    assert d.deadlines[1] == tuned.rerank_deadline_s


def test_jev_profile_unchanged() -> None:
    p = providers.openrouter("typesafe/jev-1.13")
    assert p.batch_size == 30 and p.thresholds == providers.Thresholds()


@pytest.mark.parametrize("model", MODELS)
async def test_model_thresholds_drive_bands_and_guard(make_pipeline: MakePipeline, model: str) -> None:
    thr = providers.MODEL_SETTINGS[model].thresholds
    p_mid = round((thr.rel_high + thr.rel_medium) / 2, 3)  # medium on this model's scale

    def row(url: str, content: str, engines: list[str], score: float) -> dict[str, Any]:
        return {"url": url, "title": url.split("/")[2], "content": content, "engines": engines,
                "positions": [1] * len(engines), "score": score, "category": "general"}  # fmt: skip

    def responder(params: dict[str, str]) -> dict[str, Any]:
        return {"query": params["q"], "results": [
            row("https://docs.example.com/a", "high topic", ["bing", "yandex"], 4.0),
            row("https://blog.example.com/b", "mid topic", ["bing"], 1.0),
        ], "answers": [], "corrections": [], "infoboxes": [], "suggestions": [], "unresponsive_engines": []}  # fmt: skip

    def score(qid: str, state: dict[str, Any]) -> float:
        return thr.rel_high if "high" in state["documents"][qid]["text"] else p_mid

    profile = providers.openrouter(model)
    d = SystemOneDecider(
        SystemOneClient(profile, "k", transport=httpx.MockTransport(FakeProvider(scorer=score)))
    )
    o = await make_pipeline(FakeSearx(responder=responder), decider=d).search(
        SearchParams("topic", explain=True)
    )
    bands = {r.site: r.rel for r in build_output(o).results}
    assert bands["docs.example.com"] == "high" and bands["blog.example.com"] == "medium"
