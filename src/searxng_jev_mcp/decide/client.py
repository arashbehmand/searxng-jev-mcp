"""SystemOneClient: a small httpx client for the System One wire format (`POST {base}/v1/systemone`).

Works with OpenRouter, TypeSafe direct, Ollaya and any compatible server. Keys are passed in by the
caller and never logged. One retry only on 429 or connection errors; 529/overload fails immediately;
a circuit breaker opens for 60 s after 3 consecutive failures and half-opens afterwards.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import key_transport_ok
from .providers import ProviderProfile

CIRCUIT_FAILURES = 3
CIRCUIT_OPEN_S = 60.0
LONG_OPEN_S = 3600.0  # bad key or exhausted spending limit: retrying every minute only adds latency


class DecisionError(Exception):
    """kinds: auth, quota, rate_limited, overloaded, too_large, invalid, timeout, unavailable, circuit_open,
    malformed. Messages never contain provider response text."""

    def __init__(self, kind: str, message: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass(slots=True)
class DecisionResponse:
    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any]
    model: str | None
    elapsed: float

    def noul(self, qid: str) -> float | None:
        a = self.answers.get(qid)
        if not isinstance(a, dict):
            return None
        p = a.get("noul")
        if isinstance(p, bool) or not isinstance(p, int | float):
            return None
        p = float(p)
        return p if 0.0 <= p <= 1.0 else None


def parse_error(status: int, body: Any) -> tuple[str, str]:
    """Normalise the error shapes of OpenRouter, TypeSafe, Ollaya and Vercel into (kind, message)."""
    msg = ""
    etype = ""
    if isinstance(body, dict):
        err = body.get("error")
        detail = body.get("detail")
        if isinstance(err, dict):  # OpenRouter {"error": {"message", "code"}}
            msg = str(err.get("message") or "")
        elif isinstance(err, str):  # Ollaya {error, code, detail}
            msg = err
            etype = str(body.get("code") or "")
        if isinstance(detail, dict):  # TypeSafe {"detail": {"error_type", "message"}}
            etype = str(detail.get("error_type") or etype)
            msg = msg or str(detail.get("message") or "")
        elif isinstance(detail, str):
            msg = msg or detail
        if "error_type" in body:  # Vercel {message, error_type}
            etype = str(body.get("error_type"))
            msg = msg or str(body.get("message") or "")
    elif isinstance(body, str):
        msg = body
    blob = f"{etype} {msg}".lower()
    if (
        "max_tokens_exceeded" in blob
        or "state_truncated" in blob
        or "too many tokens" in blob
        or "context length" in blob
    ):
        return "too_large", msg or etype
    if status == 402 or (status in (401, 403) and ("limit" in blob or "credit" in blob or "quota" in blob)):
        return "quota", msg or "spending limit reached"
    if status in (401, 403):
        return "auth", msg or "authentication failed"
    if status == 429:
        return "rate_limited", msg or "rate limited"
    if status == 529 or "overloaded" in blob:
        return "overloaded", msg or "provider overloaded"
    if status in (400, 404, 422):
        return "invalid", msg or f"HTTP {status}"
    return "unavailable", msg or f"HTTP {status}"


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0

    def check(self) -> None:
        if self.open_until > time.monotonic():
            raise DecisionError("circuit_open", "decision provider paused after repeated failures")

    def ok(self) -> None:
        self.failures = 0
        self.open_until = 0.0

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= CIRCUIT_FAILURES:
            self.open_until = time.monotonic() + CIRCUIT_OPEN_S
            self.failures = 0

    def open_long(self) -> None:
        self.open_until = time.monotonic() + LONG_OPEN_S
        self.failures = 0


@dataclass
class _RateLimiter:
    rate: float
    _next: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + 1.0 / self.rate


class SpendLog:
    """JSONL of per-call usage (tokens, USD cost) for budget tracking. No payloads, no keys."""

    def __init__(self, path: Path | None):
        self.path = path
        self.total_cost = 0.0
        self.calls = 0

    def write(
        self, profile: ProviderProfile, usage: dict[str, Any], elapsed: float, purpose: str, status: int
    ) -> None:
        cost = usage.get("cost")
        if isinstance(cost, int | float):
            self.total_cost += float(cost)
        self.calls += 1
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": round(time.time(), 3), "provider": profile.name, "model": profile.model,
                    "purpose": purpose, "status": status, "input_tokens": usage.get("input_tokens"),
                    "cost": cost, "s": round(elapsed, 3),
                }) + "\n")  # fmt: skip
        except OSError:
            pass


class SystemOneClient:
    def __init__(
        self,
        profile: ProviderProfile,
        api_key: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        spend: SpendLog | None = None,
        user: str | None = None,
    ):
        self.profile = profile
        self._key = api_key
        if profile.auth and api_key and not key_transport_ok(profile.base_url):
            raise ValueError(
                f"refusing to send an API key to {profile.base_url}: use https (or http only to localhost)"
            )
        headers = {"Content-Type": "application/json", **profile.extra_headers}
        if profile.auth and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=profile.base_url.rstrip("/"), headers=headers, transport=transport
        )
        self.breaker = _Breaker()
        self.limiter = _RateLimiter(profile.rate_per_s)
        self.spend = spend or SpendLog(None)
        self.user = user
        self.last_error: str | None = None
        self.last_latency: float | None = None

    def __repr__(self) -> str:  # never show the key
        return f"SystemOneClient({self.profile.label})"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - per-request HTTP timeout, passed to httpx
        purpose: str = "decide",
    ) -> DecisionResponse:
        self.breaker.check()
        body: dict[str, Any] = {"model": self.profile.model, "state": state, "questions": questions}
        if self.profile.needs_user and self.user:
            body["user"] = self.user
        timeout = timeout or self.profile.timeout_s
        attempt = 0
        while True:
            attempt += 1
            await self.limiter.wait()
            t0 = time.perf_counter()
            try:
                resp = await self._client.post("/v1/systemone", json=body, timeout=timeout)
            except httpx.TimeoutException as e:
                self.breaker.fail()
                self.last_error = "timeout"
                raise DecisionError("timeout", f"decision call exceeded {timeout:.1f}s") from e
            except httpx.HTTPError as e:
                if attempt == 1:
                    continue
                self.breaker.fail()
                self.last_error = "unavailable"
                raise DecisionError(
                    "unavailable", f"cannot reach decision provider: {type(e).__name__}"
                ) from e
            elapsed = time.perf_counter() - t0
            try:
                data = resp.json()
            except ValueError:
                data = resp.text[:300]
            if resp.status_code == 200 and isinstance(data, dict):
                raw_usage = data.get("usage")
                usage: dict[str, Any] = dict(raw_usage) if isinstance(raw_usage, dict) else {}
                if "cost" not in usage and self.profile.price_per_m_input and usage.get("input_tokens"):
                    usage["cost"] = float(usage["input_tokens"]) * self.profile.price_per_m_input / 1e6
                    usage["cost_estimated"] = True
                self.spend.write(self.profile, usage, elapsed, purpose, 200)
                answers = data.get("answers")
                if not isinstance(answers, dict):
                    self.breaker.fail()
                    self.last_error = "malformed"
                    raise DecisionError("malformed", "response has no `answers` object", 200)
                self.breaker.ok()
                self.last_error = None
                self.last_latency = elapsed
                return DecisionResponse(answers, usage, data.get("model"), elapsed)
            kind, _msg = parse_error(resp.status_code, data)
            self.spend.write(self.profile, {}, elapsed, purpose, resp.status_code)
            if kind == "rate_limited" and attempt == 1:
                retry_after = _retry_after(resp)
                if retry_after is not None and retry_after <= 1.0:
                    await asyncio.sleep(retry_after)
                    continue
                # a longer Retry-After is respected: fail now instead of hammering the provider
            if kind in ("auth", "quota"):
                self.breaker.open_long()
            elif kind not in ("too_large", "invalid"):
                self.breaker.fail()
            self.last_error = kind
            # Provider error bodies can echo account identifiers; callers only ever see kind + status.
            raise DecisionError(kind, f"{kind} (HTTP {resp.status_code})", resp.status_code)

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.profile.name,
            "model": self.profile.model,
            "key": "set" if self._key else ("not needed" if not self.profile.auth else "unset"),
            "circuit_open": self.breaker.open_until > time.monotonic(),
            "last_error": self.last_error,
            "last_latency_s": round(self.last_latency, 3) if self.last_latency else None,
            "calls": self.spend.calls,
            "cost_usd": round(self.spend.total_cost, 6),
        }


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else 0.2
    except ValueError:
        return None
