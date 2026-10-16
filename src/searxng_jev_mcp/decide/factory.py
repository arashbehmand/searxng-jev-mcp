"""Build decision backends from settings.

`auto` chains every available provider in the order OpenRouter → Ollaya → TypeSafe with runtime
failover; an explicit provider can name fallbacks in DECISION_FALLBACK. `private=true` calls use only a
local Ollaya decider (or none).
"""

from __future__ import annotations

import httpx

from ..config import Settings
from ..store import Cache
from . import providers
from .backend import ChainDecider, SystemOneDecider
from .client import SpendLog, SystemOneClient
from .providers import ProviderProfile


def ollaya_reachable(base_url: str, timeout: float = 0.3) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=timeout)
    except httpx.HTTPError:
        return False
    return r.status_code == 200


def profile_for(settings: Settings, provider: str) -> tuple[ProviderProfile | None, str | None, str | None]:
    """Return (profile, api_key, reason_if_unavailable) for one named provider."""
    model = settings.decision_model
    if provider == "openrouter":
        key = settings.key_for("openrouter")
        if not key:
            return None, None, "OPENROUTER_API_KEY is not set"
        return providers.openrouter(model or "typesafe/jev-1.13", settings.openrouter_base_url), key, None
    if provider == "typesafe":
        key = settings.key_for("typesafe")
        if not key:
            return None, None, "TYPESAFE_API_KEY is not set"
        m = model if model and "/" not in model else "jev-1.13.0"
        return providers.typesafe(m, settings.typesafe_base_url), key, None
    if provider == "ollaya":
        m = model if model and "/" not in model and ":" in model else settings.ollaya_model
        return providers.ollaya(m, settings.ollaya_base_url), None, None
    if provider == "custom":
        if not settings.decision_base_url:
            return None, None, "DECISION_BASE_URL is not set"
        key = settings.key_for("custom")
        return providers.custom(settings.decision_base_url, model or "jev-1.13.0", auth=bool(key)), key, None
    return None, None, "decisions disabled (DECISION_PROVIDER=none)"


def make_decider(
    profile: ProviderProfile, key: str | None, settings: Settings, cache: Cache | None
) -> SystemOneDecider:
    spend = SpendLog(settings.data_dir / "spend.jsonl")
    return SystemOneDecider(SystemOneClient(profile, key, spend=spend), cache)


def _chain_names(settings: Settings) -> list[str]:
    if settings.decision_provider == "auto":
        return ["openrouter", "ollaya", "typesafe"]
    names = [settings.decision_provider]
    for extra in (settings.decision_fallback or "").split(","):
        extra = extra.strip().lower()
        if extra and extra not in names and extra in ("openrouter", "typesafe", "ollaya", "custom"):
            names.append(extra)
    return names


def build_deciders(
    settings: Settings, cache: Cache | None
) -> tuple[SystemOneDecider | ChainDecider | None, SystemOneDecider | None, str | None]:
    """Return (decider, private_decider, reason_if_no_decider).

    `auto` chains every available provider (OpenRouter key → reachable Ollaya → TypeSafe key) with
    runtime failover; an explicit DECISION_PROVIDER uses that one, followed by DECISION_FALLBACK.
    """
    chain: list[SystemOneDecider] = []
    reasons: list[str] = []
    for name in _chain_names(settings):
        if name == "none":
            break
        if (
            name == "ollaya"
            and settings.decision_provider == "auto"
            and not ollaya_reachable(settings.ollaya_base_url)
        ):
            reasons.append("Ollaya not reachable")
            continue
        profile, key, why = profile_for(settings, name)
        if profile is None:
            reasons.append(why or name)
            continue
        try:
            chain.append(make_decider(profile, key, settings, cache))
        except ValueError as e:  # e.g. a key would travel over plain http
            reasons.append(str(e))
    decider: SystemOneDecider | ChainDecider | None
    if not chain:
        decider = None
    elif len(chain) == 1:
        decider = chain[0]
    else:
        decider = ChainDecider(chain)
    reason = None if decider else ("no decision provider available (" + "; ".join(reasons) + ")")

    # private=true never uses a hosted provider. The Ollaya decider is created without probing; if
    # Ollaya is not running its calls fail fast and the pipeline falls back to prior ranking.
    local = next((d for d in chain if d.profile.name == "ollaya"), None)
    if local is not None:
        private = local
    else:
        p, _, _ = profile_for(settings, "ollaya")
        private = make_decider(p, None, settings, cache) if p else None
    return decider, private, reason
