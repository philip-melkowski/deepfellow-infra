# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shared HuggingFace Hub API helpers for the model-list generator scripts."""

import asyncio
import contextlib
import json
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiohttp

HF_API = "https://huggingface.co/api"

DEFAULT_MODEL_DETAIL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "hf_models"
DEFAULT_MODEL_DETAIL_CACHE_TTL_SECONDS = 1200.0  # 20 min: covers back-to-back get-vllm-models / get-sglang-models runs in one job


class HfErrorKind(StrEnum):
    """Why a HuggingFace Hub request failed, as far as the response lets us tell."""

    NOT_FOUND = "not_found"
    REVISION_NOT_FOUND = "revision_not_found"
    GATED = "gated"
    NO_ACCESS = "no_access"
    UNAVAILABLE = "unavailable"


def classify_hf_error(exc: Exception) -> HfErrorKind:
    """Classify a failed HuggingFace request by its status and `X-Error-Code` header.

    HF answers an anonymous request for a missing *or* private repo with a bare 401 (no
    `X-Error-Code`), so as not to leak private repo names - `NO_ACCESS` therefore also covers
    "doesn't exist" when no token was sent, and callers word their message accordingly.
    Anything that isn't a definite HTTP answer (429/5xx after retries, connection errors,
    timeouts) is `UNAVAILABLE`.
    """
    if not isinstance(exc, aiohttp.ClientResponseError):
        return HfErrorKind.UNAVAILABLE
    error_code = exc.headers.get("X-Error-Code") if exc.headers else None
    if error_code == "GatedRepo":
        return HfErrorKind.GATED
    if exc.status == 404:
        return HfErrorKind.REVISION_NOT_FOUND if error_code == "RevisionNotFound" else HfErrorKind.NOT_FOUND
    if exc.status in (401, 403):
        return HfErrorKind.NO_ACCESS
    return HfErrorKind.UNAVAILABLE


async def get_json_with_retry(session: aiohttp.ClientSession, url: str, params: dict[str, str], max_retries: int = 8) -> Any:  # noqa: ANN401
    """GET url as JSON, retrying with backoff when HuggingFace rate-limits (429) or errors transiently (5xx).

    The default `max_retries`/backoff cap are sized for anonymous (unauthenticated) requests, whose
    rate-limit window is short but can still take over a minute to clear once a batch of list/detail
    calls has exhausted it — a handful of quick retries capped at 30s wasn't enough to ride that out.
    """
    for attempt in range(max_retries):
        async with session.get(url, params=params) as resp:
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError as exc:
                if (exc.status != 429 and exc.status < 500) or attempt == max_retries - 1:
                    raise
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(3.0 * 2.0**attempt, 60.0)
                await asyncio.sleep(delay)
                continue
            return await resp.json(content_type=None)
    raise AssertionError("unreachable")  # pragma: no cover


def _model_detail_cache_path(cache_dir: Path, model_id: str) -> Path:
    return cache_dir / f"{model_id.replace('/', '--')}.json"


async def get_model_detail_cached(
    session: aiohttp.ClientSession,
    model_id: str,
    max_retries: int = 8,
    cache_dir: Path | None = None,
    ttl_seconds: float = DEFAULT_MODEL_DETAIL_CACHE_TTL_SECONDS,
) -> Any:  # noqa: ANN401
    """Fetch a model's full detail payload (`?blobs=true`), optionally reusing a disk cache across script invocations.

    The vLLM and SGLang registries (`just get-vllm-models` / `get-sglang-models`) run
    get_huggingface_models.py with the same search parameters for LLMs and embeddings, so they hit
    this endpoint for a largely overlapping set of model ids. Caching by model_id lets the later run
    reuse the earlier one's response instead of re-querying HuggingFace, which is what was tripping
    its rate limit. get_llamacpp_models.py uses the same helper, but its GGUF-only model set doesn't
    overlap the HF script's (which excludes GGUF), so it only benefits from its own reruns.

    `cache_dir=None` (the default) disables caching entirely - a plain passthrough to
    `get_json_with_retry`. `fetch_registry_entries` and `fetch_llamacpp_entries` opt in via
    `DEFAULT_MODEL_DETAIL_CACHE_DIR`, so both the CLI scripts and the server's in-process catalog
    refresh use the cache; direct callers of `fetch_model_details` / `fetch_model_files` (e.g. tests)
    stay free of disk side effects.

    One file per model_id (rather than a single shared cache file) avoids read-modify-write races when
    multiple `just get-*-models` invocations run concurrently; a cache miss just costs a duplicate fetch.
    """
    if cache_dir is None:
        return await get_json_with_retry(session, f"{HF_API}/models/{model_id}", {"blobs": "true"}, max_retries=max_retries)

    path = _model_detail_cache_path(cache_dir, model_id)
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - cached["fetched_at"] < ttl_seconds:
                return cached["data"]
        except (ValueError, KeyError, TypeError, OSError):  # ValueError covers JSONDecodeError
            pass  # corrupt or unreadable cache entry - fall through and refetch

    data = await get_json_with_retry(session, f"{HF_API}/models/{model_id}", {"blobs": "true"}, max_retries=max_retries)

    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps({"fetched_at": time.time(), "data": data}), encoding="utf-8")
        tmp_path.replace(path)  # atomic rename - safe if another process is writing the same model_id concurrently
    except OSError:
        # the cache is only an optimization - never fail a successful fetch over it
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)

    return data


def _has_native_weights(siblings: list[dict[str, Any]]) -> bool:
    """Return True if the repo root holds weights vLLM can load directly (safetensors or PyTorch `.bin`)."""
    for f in siblings:
        filename = f.get("rfilename", "").lower()
        if "/" in filename:
            continue  # only the repo root counts - a nested gguf/ or onnx/ folder is not what the engine loads
        if filename.endswith((".safetensors", ".safetensors.index.json")) or (
            filename.startswith("pytorch_model") and filename.endswith(".bin")
        ):
            return True
    return False


def is_gguf(model: dict[str, Any]) -> bool:
    """Return True if the model repo looks like a GGUF (quantized) repo.

    Repos that also ship native weights at their root (multi-format repos such as mixedbread-ai/mxbai-embed-large-v1)
    are not GGUF-only, even if they carry the "gguf" tag. When the payload has no file list (e.g. search results),
    this falls back to the name/tag check alone.
    """
    if _has_native_weights(model.get("siblings") or []):
        return False
    name = model.get("id", "").lower()
    tags = [t.lower() for t in model.get("tags", [])]
    return "gguf" in name or "gguf" in tags


EMBEDDING_RE = re.compile(
    r"embed"  # generic
    r"|bge-"  # BAAI
    r"|gte-"  # thenlper/Alibaba-NLP
    r"|(?:^|[-/])e5-(?:large|base|small|mini)",  # intfloat; anchored + size suffix, "e5-" alone is too
    # common a substring (e.g. "...-Fable5-Thinking-...") to match unanchored.
    re.IGNORECASE,
)


def is_embedding(model: dict[str, Any]) -> bool:
    """Filter embedding models by name heuristic.

    Covers the generic "embed" substring plus the common family prefixes (BAAI's bge-,
    thenlper/Alibaba's gte-, intfloat's e5-) that ship without "embed" anywhere in the repo name.
    """
    return bool(EMBEDDING_RE.search(model.get("id", "")))
