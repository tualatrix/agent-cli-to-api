from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_CODEX_ADVERTISED_MODELS, settings

logger = logging.getLogger("uvicorn.error")

CACHE_TTL_S = 300
FETCH_TIMEOUT_S = 20

DEFAULT_CLAUDE_MODELS = [
    "sonnet",
    "opus",
    "haiku",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-5",
    "claude-haiku-4-5-20251001",
]

DEFAULT_GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

DEFAULT_CURSOR_MODELS = [
    "auto",
    "gpt-5.3-codex",
    "gpt-5.3-codex-fast",
    "gpt-5.3-codex-high",
    "composer-2.5",
    "composer-2.5-fast",
    "gpt-5.2",
    "gpt-5.6-sol-high",
    "claude-sonnet-5-thinking-high",
    "claude-opus-5-thinking-high",
    "gemini-3.7-flash-high",
    "cursor-grok-4.6-high",
]

_CACHE: dict[str, tuple[float, list[str]]] = {}
_LAST_GOOD: dict[str, list[str]] = {}
_CACHE_LOCK: asyncio.Lock | None = None


def _ensure_lock() -> asyncio.Lock:
    global _CACHE_LOCK
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()
    return _CACHE_LOCK


def clear_model_cache() -> None:
    _CACHE.clear()
    _LAST_GOOD.clear()


def fallback_models_for(provider: str) -> list[str]:
    if provider in {"auto", "codex"}:
        return list(DEFAULT_CODEX_ADVERTISED_MODELS)
    if provider == "claude":
        return list(DEFAULT_CLAUDE_MODELS)
    if provider == "gemini":
        return list(DEFAULT_GEMINI_MODELS)
    if provider == "cursor-agent":
        return list(DEFAULT_CURSOR_MODELS)
    return []


def known_models(provider: str) -> list[str]:
    cached = _CACHE.get(provider)
    if cached:
        return list(cached[1])
    last_good = _LAST_GOOD.get(provider)
    if last_good:
        return list(last_good)
    return fallback_models_for(provider)


def _dedupe(models: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in models:
        model_id = (raw or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    return out


def _prefix_model(provider: str, model_id: str) -> str:
    if provider == "codex":
        return model_id
    if provider == "cursor-agent":
        return f"cursor:{model_id}"
    return f"{provider}:{model_id}"


def parse_cursor_model_list(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if raw[0] in "[{":
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None
        if payload is not None:
            return _model_ids_from_payload(payload)
    models: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("available models"):
            continue
        model_id = line.split(" - ", 1)[0].strip()
        if model_id:
            models.append(model_id)
    return _dedupe(models)


def _model_ids_from_payload(payload: Any) -> list[str]:
    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("data", "models", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
        else:
            items = [payload]
    else:
        return []

    models: list[str] = []
    for item in items:
        if isinstance(item, str):
            models.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("id", "name", "model", "modelId", "model_id"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                models.append(value.strip())
                break
    return _dedupe(_strip_resource_prefix(m) for m in models)


def _strip_resource_prefix(model_id: str) -> str:
    raw = (model_id or "").strip()
    if raw.startswith("models/"):
        return raw.split("/", 1)[1]
    return raw


def _is_gemini_chat_model(model_id: str) -> bool:
    name = (model_id or "").strip().lower()
    if not name.startswith("gemini"):
        return False
    skip = ("embed", "image", "imagen", "aqa", "tts", "robotics", "computer-use")
    return not any(token in name for token in skip)


async def list_models_for_provider(provider: str) -> list[str]:
    now = time.time()
    cached = _CACHE.get(provider)
    if cached and cached[0] > now:
        return list(cached[1])

    lock = _ensure_lock()
    async with lock:
        cached = _CACHE.get(provider)
        if cached and cached[0] > now:
            return list(cached[1])
        try:
            fetched = await _fetch_models_for_provider(provider)
        except Exception as exc:
            logger.warning("[models] %s live fetch failed: %s", provider, exc)
            fetched = []
        live = _dedupe(fetched or [])
        models = _dedupe([*live, *fallback_models_for(provider)])
        ttl = CACHE_TTL_S if live else 30
        _CACHE[provider] = (now + ttl, models)
        if live:
            _LAST_GOOD[provider] = list(models)
            logger.info("[models] %s listed %d models", provider, len(models))
        return list(models)


async def collect_advertised_models(provider: str, *, default_id: str) -> list[str]:
    if provider == "auto":
        groups: list[list[str]] = [["default", default_id]]
        for item in ("codex", "cursor-agent", "claude", "gemini"):
            if provider_auth_ready(item):
                models = await list_models_for_provider(item)
            else:
                models = fallback_models_for(item)
            if item == "codex":
                groups.append(models)
            else:
                groups.append([_prefix_model(item, model_id) for model_id in models])
        return _dedupe([model_id for group in groups for model_id in group])

    models = await list_models_for_provider(provider)
    return _dedupe(["default", default_id, *models])


async def warmup_models(provider: str) -> None:
    if provider == "auto":
        await asyncio.gather(
            *(
                list_models_for_provider(item)
                for item in ("codex", "cursor-agent", "claude", "gemini")
                if provider_auth_ready(item)
            ),
            return_exceptions=True,
        )
        return
    await list_models_for_provider(provider)


async def _fetch_models_for_provider(provider: str) -> list[str]:
    if provider == "codex":
        return list(DEFAULT_CODEX_ADVERTISED_MODELS)
    if provider == "cursor-agent":
        return await _fetch_cursor_models()
    if provider == "claude":
        return await _fetch_claude_models()
    if provider == "gemini":
        return await _fetch_gemini_models()
    return []


async def _fetch_cursor_models() -> list[str]:
    bin_path = shutil.which(settings.cursor_agent_bin) or settings.cursor_agent_bin
    if not bin_path:
        return []
    cmd = [bin_path]
    if settings.cursor_agent_extra_args:
        cmd.extend(settings.cursor_agent_extra_args)
    if settings.cursor_agent_api_key:
        cmd.extend(["--api-key", settings.cursor_agent_api_key])
    cmd.append("--list-models")
    env = os.environ.copy()
    if settings.cursor_agent_api_key:
        env.setdefault("CURSOR_API_KEY", settings.cursor_agent_api_key)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FETCH_TIMEOUT_S)
    except asyncio.TimeoutError:
        with suppress(Exception):
            proc.kill()
        raise RuntimeError("cursor-agent --list-models timed out")
    text = (stdout or b"").decode("utf-8", errors="ignore")
    models = parse_cursor_model_list(text)
    if models:
        return models
    err = (stderr or b"").decode("utf-8", errors="ignore").strip()
    if proc.returncode:
        raise RuntimeError(err or f"cursor-agent --list-models exited {proc.returncode}")
    return []


async def _fetch_claude_models() -> list[str]:
    from .claude_oauth import list_oauth_models

    return await list_oauth_models(timeout_seconds=min(settings.timeout_seconds, FETCH_TIMEOUT_S))


async def _fetch_gemini_models() -> list[str]:
    from .gemini_cloudcode import list_cloudcode_models

    models = await list_cloudcode_models(timeout_seconds=min(settings.timeout_seconds, FETCH_TIMEOUT_S))
    return [model_id for model_id in models if _is_gemini_chat_model(model_id)]


def provider_auth_ready(provider: str) -> bool:
    if provider == "codex":
        home = Path(settings.codex_cli_home) if settings.codex_cli_home else Path.home()
        return (home / ".codex" / "auth.json").exists() or (Path.home() / ".codex" / "auth.json").exists()
    if provider == "cursor-agent":
        return bool(shutil.which(settings.cursor_agent_bin) or Path(settings.cursor_agent_bin).expanduser().exists())
    if provider == "claude":
        oauth = Path(settings.claude_oauth_creds_path).expanduser()
        cli_settings = Path.home() / ".claude" / "settings.json"
        return oauth.exists() or cli_settings.exists()
    if provider == "gemini":
        return Path(settings.gemini_oauth_creds_path).expanduser().exists()
    return False
