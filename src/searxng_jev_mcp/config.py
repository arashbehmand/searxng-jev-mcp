"""Settings: environment variables, explicit env files and the OS keychain.

Sources, highest precedence first:

1. the process environment (e.g. ``claude mcp add -e SEARXNG_URL=...``);
2. env files named explicitly: ``--env-file PATH`` on the server command line (repeatable) and
   ``$SEARXNG_JEV_ENV_FILE`` (one path, or several separated by ``os.pathsep``); later files win;
3. ``~/.config/searxng-jev-mcp/env``.

Files in the current working directory are **not** read implicitly: when the server runs from an
unrelated project (``uvx``, an installed script), a ``.env`` there could otherwise point the API key
at another host. API keys missing from all sources are looked up in the OS keychain (service
``searxng-jev-mcp``, account ``openrouter`` / ``typesafe`` / ``custom``). Secret values are held as
``SecretStr`` and never logged; use :func:`describe` for diagnostics.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openrouter", "ollaya", "typesafe", "custom", "none", "auto"]
GuardMode = Literal["off", "pin3", "pin3_p80", "pin3_band", "pin1_p80", "pin1_band", "band_reorder"]

KEYRING_SERVICE = "searxng-jev-mcp"
# Sent as OpenRouter's HTTP-Referer attribution header. Forks should point it at their own repository
# or set it to None, which sends no Referer.
PROJECT_URL: str | None = "https://github.com/arashbehmand/searxng-jev-mcp"
USER_CONFIG_FILE = Path.home() / ".config" / "searxng-jev-mcp" / "env"
REPO_ROOT = Path(__file__).resolve().parents[2]

_cli_env_files: list[Path] = []


def set_cli_env_files(paths: Sequence[str | Path]) -> None:
    """Record ``--env-file`` arguments (resolved against the current directory at start-up)."""
    _cli_env_files[:] = [Path(p).expanduser().resolve() for p in paths]
    get_settings.cache_clear()


def default_env_files() -> list[Path]:
    files: list[Path] = [USER_CONFIG_FILE]
    extra = os.environ.get("SEARXNG_JEV_ENV_FILE")
    if extra:
        files += [Path(p).expanduser() for p in extra.split(os.pathsep) if p.strip()]
    files += _cli_env_files
    return files


def repo_env_files() -> list[Path]:
    """For the repository's own dev scripts and eval: its ``.env`` and ``.env.local`` (explicit)."""
    return [REPO_ROOT / ".env", REPO_ROOT / ".env.local"]


def _default_data_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "searxng-jev-mcp"


def key_transport_ok(base_url: str) -> bool:
    """API keys only travel over HTTPS, or plain HTTP to this machine."""
    parts = urlsplit(base_url)
    return parts.scheme == "https" or (
        parts.scheme == "http" and (parts.hostname or "") in ("127.0.0.1", "localhost", "::1")
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,  # files are passed explicitly by load_settings()
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- SearXNG -------------------------------------------------------------------------------
    searxng_url: str | None = Field(default=None, validation_alias="SEARXNG_URL")
    searxng_username: str | None = Field(default=None, validation_alias="SEARXNG_USERNAME")
    searxng_password: SecretStr | None = Field(default=None, validation_alias="SEARXNG_PASSWORD")
    # Concurrent requests to SearXNG. A search makes 1-2 requests (general + verticals), so 4 allows the
    # 2-4 concurrent searches without starving the vertical request.
    searxng_max_concurrency: int = Field(default=4, ge=1, le=8, validation_alias="SEARXNG_MAX_CONCURRENCY")
    # Google CSE stops working on 2027-01-01; it is excluded from the baseline after that date.
    enable_google_cse: bool = Field(default=True, validation_alias="ENABLE_GOOGLE_CSE")
    google_cse_sunset: dt.date = dt.date(2027, 1, 1)

    # --- Decisions -----------------------------------------------------------------------------
    decision_provider: ProviderName = Field(default="auto", validation_alias="DECISION_PROVIDER")
    decision_model: str | None = Field(
        default=None, validation_alias=AliasChoices("DECISION_MODEL", "TYPESAFE_DEFAULT_MODEL")
    )
    decision_base_url: str | None = Field(default=None, validation_alias="DECISION_BASE_URL")
    # Runtime fallback chain after the primary provider, e.g. "typesafe" or "typesafe,ollaya".
    # DECISION_PROVIDER=auto implies "every available provider in the order openrouter, typesafe".
    decision_fallback: str | None = Field(default=None, validation_alias="DECISION_FALLBACK")
    # Stage B: let the decision model choose the vertical engines (balanced mode). Off by default: the
    # end-to-end eval found no gain over the static keyword map or the general engines alone, while it
    # adds a serial step before the vertical request (docs/evaluation.md). DECISION_ROUTING=1 turns it on.
    decision_routing: bool = Field(default=False, validation_alias="DECISION_ROUTING")
    # Authority guard after the rerank (pipeline.apply_guard). "pin3" was chosen on the dev split; see
    # docs/evaluation.md for every variant, including the test-split results that favour "pin3_p80".
    authority_guard: GuardMode = Field(default="pin3_p80", validation_alias="AUTHORITY_GUARD")
    decision_api_key: SecretStr | None = Field(default=None, validation_alias="DECISION_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api", validation_alias="OPENROUTER_BASE_URL"
    )
    typesafe_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("TYPESAFE_API_KEY", "TYPESAFE_AI_API_KEY")
    )
    typesafe_base_url: str = Field(default="https://api.typesafe.ai", validation_alias="TYPESAFE_BASE_URL")
    ollaya_base_url: str = Field(default="http://127.0.0.1:11435", validation_alias="OLLAYA_BASE_URL")
    ollaya_model: str = Field(default="winnow:e4b", validation_alias="OLLAYA_MODEL")

    # --- Behaviour -----------------------------------------------------------------------------
    debug_tools: bool = Field(default=False, validation_alias="DEBUG_TOOLS")
    engine_overrides: str | None = Field(
        default=None, validation_alias="ENGINE_OVERRIDES"
    )  # "yep=off,duckduckgo=on"
    data_dir: Path = Field(default_factory=_default_data_dir, validation_alias="SEARXNG_JEV_DATA_DIR")
    cache_enabled: bool = Field(default=True, validation_alias="SEARXNG_JEV_CACHE")
    trace_enabled: bool = Field(default=True, validation_alias="SEARXNG_JEV_TRACE")
    keyring_enabled: bool = Field(default=True, validation_alias="SEARXNG_JEV_KEYRING")
    log_level: str = Field(default="WARNING", validation_alias="SEARXNG_JEV_LOG_LEVEL")

    @field_validator("searxng_url", "decision_base_url", mode="after")
    @classmethod
    def _strip_slash(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    # --- Keys ----------------------------------------------------------------------------------
    def key_for(self, provider: str) -> str | None:
        """Return the API key for a provider (env/.env first, then the OS keychain)."""
        field = {
            "openrouter": self.openrouter_api_key,
            "typesafe": self.typesafe_api_key,
            "custom": self.decision_api_key,
        }.get(provider)
        if field is not None and field.get_secret_value():
            return field.get_secret_value()
        if provider in ("openrouter", "typesafe", "custom") and self.keyring_enabled:
            return _keyring_get(provider)
        return None

    def google_cse_active(self, today: dt.date | None = None) -> bool:
        return self.enable_google_cse and (today or dt.date.today()) < self.google_cse_sunset


def _keyring_get(account: str) -> str | None:
    try:
        import keyring

        value = keyring.get_password(KEYRING_SERVICE, account)
    except Exception:  # no backend, locked keychain, etc.
        return None
    return value or None


def mask(secret: str | None) -> str:
    """Describe a secret without revealing any of it (not even a prefix)."""
    if not secret:
        return "unset"
    return f"set ({len(secret)} chars)"


def describe(settings: Settings) -> dict[str, str]:
    """Diagnostics-safe view of the settings."""
    return {
        "searxng_url": "set" if settings.searxng_url else "unset",
        "decision_provider": settings.decision_provider,
        "decision_model": settings.decision_model or "(provider default)",
        "openrouter_api_key": mask(settings.key_for("openrouter")),
        "typesafe_api_key": mask(settings.key_for("typesafe")),
        "data_dir": "set",
    }


def load_settings(env_files: Sequence[str | Path] | None = None) -> Settings:
    """Build settings from the environment plus the given env files (missing files are skipped)."""
    files = tuple(Path(f) for f in (env_files if env_files is not None else default_env_files()))
    return Settings(_env_file=files or None)  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
