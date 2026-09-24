"""OpenAI-compatible LLM client wrapper.

Every model in the registry is queried through the OpenAI Chat Completions API using its
own base_url + api_key, so adding a provider is purely a config change. Provides:
  - automatic retry with exponential backoff on transient errors (tenacity)
  - a uniform LLMResult (text, usage, finish_reason, ok/error) so a single bad call never
    crashes a stage
  - optional strict JSON parsing helper for judge/generation prompts
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import Config, ResolvedModel

# Exceptions worth retrying (transient): timeouts, connection drops, 429s, and 5xx server errors.
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


@dataclass
class LLMResult:
    """Uniform result of one chat completion attempt.

    Token counts come straight from the provider's `usage` field — these are the ACTUAL counts
    the provider reports/bills, not a local estimate. `reasoning_tokens` is populated for
    reasoning models that expose it; it is a subset of `completion_tokens`.
    """

    text: str
    model_key: str
    model_id: str
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    ok: bool = True
    error: str = ""


@dataclass
class LLMClient:
    """Caches one OpenAI client per model key and runs completions with retry."""

    cfg: Config
    _clients: dict[str, OpenAI] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _client_for(self, m: ResolvedModel) -> OpenAI:
        with self._lock:
            if m.key not in self._clients:
                self._clients[m.key] = OpenAI(
                    base_url=m.base_url,
                    api_key=m.api_key,
                    timeout=float(self.cfg.get("defaults", "request_timeout", default=60)),
                    max_retries=0,  # we manage retries ourselves via tenacity
                )
            return self._clients[m.key]

    def _effective_temperature(self, m: ResolvedModel, requested: float | None) -> float | None:
        """Resolve the temperature to send, or None to omit it entirely.

        Precedence: a model that opts out (`omit_temperature`) sends nothing; a model with a
        fixed `temperature` always sends that; otherwise we send the caller's requested value,
        and `None` means "don't send temperature" so the model uses its own default.
        """
        if m.omit_temperature:
            return None
        if m.temperature is not None:
            return m.temperature
        return requested

    def complete(
        self,
        model_key: str,
        *,
        user: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Run one completion. Never raises for model/API errors — returns ok=False instead.

        `temperature=None` (the default) omits the parameter so the model uses its own default.
        `max_tokens=None` (the default) omits the output cap so each model can produce its full
        answer / reasoning — important since models reason for differing lengths.
        """
        m = self.cfg.model(model_key)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        eff_temp = self._effective_temperature(m, temperature)

        retry_cfg = self.cfg.get("retry", default={}) or {}
        attempts = int(retry_cfg.get("max_attempts", 5))
        backoff = float(retry_cfg.get("backoff_seconds", 2))
        max_backoff = float(retry_cfg.get("max_backoff", 60))

        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            wait=wait_random_exponential(multiplier=backoff, max=max_backoff),
            stop=stop_after_attempt(attempts),
            reraise=True,
        )
        def _call() -> Any:
            kwargs: dict[str, Any] = {"model": m.model, "messages": messages}
            if eff_temp is not None:
                kwargs["temperature"] = eff_temp
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return self._client_for(m).chat.completions.create(**kwargs)

        try:
            resp = _call()
        except (APIStatusError, *_RETRYABLE) as exc:  # exhausted retries or non-retryable status
            return LLMResult("", model_key, m.model, ok=False, error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - last-resort guard so a stage never dies
            return LLMResult("", model_key, m.model, ok=False, error=f"{type(exc).__name__}: {exc}")

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        prompt_tokens = _usage_get(usage, "prompt_tokens")
        completion_tokens = _usage_get(usage, "completion_tokens")
        total_tokens = _usage_get(usage, "total_tokens") or (prompt_tokens + completion_tokens)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = _usage_get(details, "reasoning_tokens")
        return LLMResult(
            text=(choice.message.content or "").strip(),
            model_key=model_key,
            model_id=m.model,
            finish_reason=getattr(choice, "finish_reason", "") or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
        )


def _usage_get(obj: Any, name: str) -> int:
    """Read a token count from a usage object/dict, tolerating either shape. 0 if absent."""
    if obj is None:
        return 0
    if isinstance(obj, dict):
        return int(obj.get(name) or 0)
    return int(getattr(obj, name, 0) or 0)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from a model reply (handles code fences / prose)."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).rsplit("```", 1)[0].strip()
    candidates = [cleaned]
    match = _JSON_BLOCK.search(cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
