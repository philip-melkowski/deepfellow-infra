# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for get_llamacpp_models.py script."""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from scripts.get_llamacpp_models import (
    build_entries,
    collect_gguf_models,
    fetch_llamacpp_entries,
    fetch_model_files,
    fetch_popular_gguf_models,
    fmt_size,
    fmt_size_compact,
    get_json_with_retry,
    is_embedding,
    is_gguf,
    is_gguf_file,
    is_llm,
    is_mmproj_file,
    is_split_gguf_file,
    main,
    quant_suffix,
)
from scripts.utils.hf_common import DEFAULT_MODEL_DETAIL_CACHE_DIR


@pytest.mark.parametrize(
    ("size_bytes", "expected_output"),
    [
        (512, "512.0 B"),
        (1000, "1.0 KB"),
        (1000**3, "1.0 GB"),
        (1000**5, "1.0 PB"),
    ],
)
def test_fmt_size_variants(size_bytes: int, expected_output: str) -> None:
    assert fmt_size(size_bytes) == expected_output


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        ("N/A", "N/A"),
        ("2.0 GB", "2.0GB"),
        ("512.0 MB", "512.0MB"),
        ("2.94 GB", "2.9GB"),
        ("2.1 GB", "2.1GB"),
        ("broken", "broken"),
    ],
)
def test_fmt_size_compact(input_val: str, expected: str) -> None:
    assert fmt_size_compact(input_val) == expected


@pytest.mark.parametrize(
    ("model_data", "expected"),
    [
        ({"id": "bartowski/Model-GGUF"}, True),
        ({"id": "org/model", "tags": ["gguf"]}, True),
        ({"id": "org/model"}, False),
        ({}, False),
        ({"id": "org/model", "tags": ["gguf"], "siblings": [{"rfilename": "model.safetensors"}, {"rfilename": "gguf/model.gguf"}]}, False),
        ({"id": "org/model", "tags": ["gguf"], "siblings": [{"rfilename": "model.safetensors.index.json"}]}, False),
        ({"id": "org/model", "tags": ["gguf"], "siblings": [{"rfilename": "pytorch_model.bin"}]}, False),
        ({"id": "org/model-GGUF", "tags": ["gguf"], "siblings": [{"rfilename": "model-Q4_K_M.gguf"}, {"rfilename": "README.md"}]}, True),
        ({"id": "org/model-GGUF", "tags": ["gguf"], "siblings": [{"rfilename": "onnx/model.safetensors"}]}, True),
    ],
    ids=[
        "gguf_in_name",
        "gguf_tag",
        "no_gguf",
        "missing_id",
        "mixed_safetensors",
        "mixed_safetensors_index",
        "mixed_pytorch_bin",
        "gguf_only_repo",
        "nested_weights_ignored",
    ],
)
def test_is_gguf(model_data: dict[str, object], expected: bool) -> None:
    assert is_gguf(model_data) is expected


@pytest.mark.parametrize(
    ("model_data", "expected"),
    [
        ({"id": "Aashraf995/KaLM-embedding-multilingual-mini-instruct-v2.5-GGUF"}, True),
        ({"id": "unsloth/bge-small-en-v1.5-GGUF"}, True),
        ({"id": "thenlper/gte-large-GGUF"}, True),
        ({"id": "unsloth/multilingual-e5-large-GGUF"}, True),
        ({"id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"}, False),
        # "e5-" is a common substring outside the intfloat family (version-ish tokens like
        # "Fable5-") and must not match without a recognized size suffix following it.
        ({"id": "GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-GGUF"}, False),
        ({}, False),
    ],
    ids=["embed_in_name", "bge_prefix", "gte_prefix", "e5_prefix", "no_embed", "e5_substring_false_positive", "missing_id"],
)
def test_is_embedding(model_data: dict[str, object], expected: bool) -> None:
    assert is_embedding(model_data) is expected


@pytest.mark.parametrize(
    ("model_data", "expected"),
    [
        ({"id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"}, True),
        ({"id": "city96/FLUX.1-dev-gguf"}, False),
        ({"id": "city96/Qwen-Image-gguf"}, False),
        ({"id": "bullerwins/Wan2.2-I2V-A14B-GGUF"}, False),
        ({"id": "Serveurperso/Qwen3-TTS-GGUF"}, False),
        ({"id": "handy-computer/whisper-large-v3-gguf"}, False),
        ({"id": "handy-computer/Qwen3-ASR-0.6B-gguf"}, False),
        ({"id": "PaddlePaddle/PaddleOCR-VL-1.6-GGUF"}, False),
        # "imatrix" is a calibrated-quant technique, not a model kind — legitimate chat LLMs
        # advertise it in their name and must not be dropped for it.
        ({"id": "DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE-Imatrix-MAX-GGUF"}, True),
        ({}, True),
    ],
    ids=["llm", "flux", "image", "video", "tts", "whisper", "asr", "ocr", "imatrix_not_excluded", "missing_id"],
)
def test_is_llm(model_data: dict[str, object], expected: bool) -> None:
    assert is_llm(model_data) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model-00001-of-00005.gguf", True),
        ("model.gguf", False),
        ("model-Q4_K_M.gguf", False),
    ],
)
def test_is_split_gguf_file(filename: str, expected: bool) -> None:
    assert is_split_gguf_file(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("mmproj-F16.gguf", True),
        ("mmproj-model-f16.gguf", True),
        ("model-Q4_K_M.gguf", False),
    ],
)
def test_is_mmproj_file(filename: str, expected: bool) -> None:
    assert is_mmproj_file(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model-Q4_K_M.gguf", True),
        ("model-00001-of-00005.gguf", False),
        ("config.json", False),
        ("model.safetensors", False),
        ("mmproj-F16.gguf", False),
        ("unet/Model-Q4_K_M.gguf", False),
        ("text_encoders/model.gguf", False),
    ],
)
def test_is_gguf_file(filename: str, expected: bool) -> None:
    assert is_gguf_file(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "q4-k-m"),
        ("mistral-community_pixtral-12b-Q5_K_M.gguf", "q5-k-m"),
        ("model.Q8_0.gguf", "q8-0"),
        ("gemma-3-270m-it-F16.gguf", "f16"),
        ("gemma-2b.gguf", "gemma-2b"),
        ("Model-IQ3_XXS.gguf", "iq3-xxs"),
        ("model-F32.gguf", "f32"),
        ("model-BF16.gguf", "bf16"),
    ],
    ids=["q4", "q5", "q8", "f16", "no_quant_token", "i_quant", "f32", "bf16"],
)
def test_quant_suffix(filename: str, expected: str) -> None:
    assert quant_suffix(filename) == expected


def test_build_entries_one_per_quant_file() -> None:
    siblings = [
        {"rfilename": "model-Q4_K_M.gguf", "size": 1024**3},
        {"rfilename": "model-Q8_0.gguf", "size": 2 * 1024**3},
        {"rfilename": "config.json", "size": 100},
    ]

    entries = build_entries("org/model", siblings)

    names = [e.name for e in entries]
    assert names == ["org/model-q4-k-m", "org/model-q8-0"]
    assert entries[0].url == "https://huggingface.co/org/model/resolve/main/model-Q4_K_M.gguf"
    assert entries[0].size == "1.1GB"
    assert entries[1].size == "2.1GB"


def test_build_entries_skips_mmproj_files() -> None:
    siblings = [
        {"rfilename": "mmproj-F16.gguf", "size": 1024**3},
        {"rfilename": "model-Q4_K_M.gguf", "size": 1024**3},
    ]

    entries = build_entries("org/model", siblings)

    names = [e.name for e in entries]
    assert names == ["org/model-q4-k-m"]


def test_build_entries_records_mmproj_presence_as_hint_not_supports_vision() -> None:
    """A companion mmproj file is a catalog hint only - llama.cpp installs don't wire it up.

    `supports_vision` must stay None (never a wrong "vision works" guess); the presence signal
    is instead carried separately as `has_mmproj`.
    """
    siblings = [
        {"rfilename": "mmproj-F16.gguf", "size": 1024**3},
        {"rfilename": "model-Q4_K_M.gguf", "size": 1024**3},
    ]

    entries = build_entries("org/model", siblings)

    assert len(entries) == 1
    assert entries[0].supports_vision is None
    assert entries[0].has_mmproj is True


def test_build_entries_has_mmproj_none_when_no_mmproj_file() -> None:
    siblings = [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    entries = build_entries("org/model", siblings)

    assert len(entries) == 1
    assert entries[0].supports_vision is None
    assert entries[0].has_mmproj is None


def test_build_entries_skips_subdirectory_files() -> None:
    siblings = [
        {"rfilename": "unet/Model-Q4_K_M.gguf", "size": 1024**3},
        {"rfilename": "text_encoders/Model-F16.gguf", "size": 1024**3},
        {"rfilename": "model-Q4_K_M.gguf", "size": 1024**3},
    ]

    entries = build_entries("org/model", siblings)

    names = [e.name for e in entries]
    assert names == ["org/model-q4-k-m"]


def test_build_entries_skips_split_files() -> None:
    siblings = [
        {"rfilename": "model-00001-of-00002.gguf", "size": 1024**3},
        {"rfilename": "model-00002-of-00002.gguf", "size": 1024**3},
    ]

    entries = build_entries("org/model", siblings)

    assert entries == []


def test_build_entries_disambiguates_colliding_quant_suffixes() -> None:
    siblings = [
        {"rfilename": "Model-NEO-IQ2_M.gguf", "size": 1024**3},
        {"rfilename": "Model-NEO-MTP-IQ2_M.gguf", "size": 2 * 1024**3},
    ]

    entries = build_entries("org/model", siblings)

    names = [e.name for e in entries]
    assert names == ["org/model-model-neo-iq2-m", "org/model-model-neo-mtp-iq2-m"]
    assert len(set(names)) == 2


def test_build_entries_no_gguf_files() -> None:
    siblings = [{"rfilename": "config.json", "size": 100}]

    entries = build_entries("org/model", siblings)

    assert entries == []


def test_build_entries_missing_size_is_na() -> None:
    siblings = [{"rfilename": "model-Q4_K_M.gguf"}]

    entries = build_entries("org/model", siblings)

    assert entries[0].size == "N/A"


def _make_mock_session(json_data: object) -> MagicMock:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


@pytest.mark.asyncio
async def test_fetch_popular_gguf_models_passes_search_param() -> None:
    session = _make_mock_session([])

    await fetch_popular_gguf_models(session, "downloads", 50)

    call_kwargs = session.get.call_args
    params = call_kwargs[1]["params"]
    assert params["search"] == "GGUF"
    assert params["sort"] == "downloads"
    assert params["limit"] == "50"


@pytest.mark.asyncio
async def test_get_json_with_retry_retries_on_429_then_succeeds() -> None:
    failing_resp = AsyncMock()
    failing_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=429))
    failing_resp.headers = {}
    failing_resp.__aenter__ = AsyncMock(return_value=failing_resp)
    failing_resp.__aexit__ = AsyncMock(return_value=False)

    ok_resp = AsyncMock()
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json = AsyncMock(return_value=[{"id": "org/model-GGUF"}])
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(side_effect=[failing_resp, ok_resp])

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await get_json_with_retry(session, "https://huggingface.co/api/models", {})

    assert result == [{"id": "org/model-GGUF"}]
    assert mock_sleep.await_count == 1


def _make_failing_resp(status: int, headers: dict[str, str] | None = None) -> AsyncMock:
    resp = AsyncMock()
    resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=status))
    resp.headers = headers or {}
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@pytest.mark.asyncio
async def test_get_json_with_retry_retries_on_5xx_then_succeeds() -> None:
    failing_resp = _make_failing_resp(503)

    ok_resp = AsyncMock()
    ok_resp.raise_for_status = MagicMock()
    ok_resp.json = AsyncMock(return_value=[{"id": "org/model-GGUF"}])
    ok_resp.__aenter__ = AsyncMock(return_value=ok_resp)
    ok_resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(side_effect=[failing_resp, ok_resp])

    with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await get_json_with_retry(session, "https://huggingface.co/api/models", {})

    assert result == [{"id": "org/model-GGUF"}]
    assert mock_sleep.await_count == 1


@pytest.mark.asyncio
async def test_get_json_with_retry_raises_after_exhausting_retries() -> None:
    session = MagicMock()
    session.get = MagicMock(side_effect=[_make_failing_resp(429) for _ in range(3)])

    with patch("asyncio.sleep", AsyncMock()), pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await get_json_with_retry(session, "https://huggingface.co/api/models", {}, max_retries=3)

    assert exc_info.value.status == 429
    assert session.get.call_count == 3


@pytest.mark.asyncio
async def test_get_json_with_retry_raises_valueerror_on_http_date_retry_after() -> None:
    failing_resp = _make_failing_resp(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    session = MagicMock()
    session.get = MagicMock(return_value=failing_resp)

    with pytest.raises(ValueError, match="Wed, 21 Oct 2015"):
        await get_json_with_retry(session, "https://huggingface.co/api/models", {}, max_retries=3)


@pytest.mark.asyncio
async def test_collect_gguf_models_deduplicates_and_filters() -> None:
    gguf_model = {"id": "org/model-GGUF"}
    non_gguf_model = {"id": "org/plain-model"}
    embedding_model = {"id": "org/embed-model-GGUF"}
    non_llm_model = {"id": "org/FLUX.1-dev-GGUF"}

    async def fake_fetch(session: Mock, sort: str, limit: int) -> list[dict[str, str]]:
        return [gguf_model, non_gguf_model, embedding_model, non_llm_model]

    with patch("scripts.get_llamacpp_models.fetch_popular_gguf_models", side_effect=fake_fetch):
        result = await collect_gguf_models(MagicMock(), {"downloads": 10})

    ids = [m["id"] for m in result]
    assert ids == ["org/model-GGUF"]


@pytest.mark.asyncio
async def test_collect_gguf_models_deduplicates_across_multiple_sort_keys() -> None:
    """Mirrors the justfile's real invocation, which combines --top-by-downloads and --top-by-likes."""
    shared_model = {"id": "org/shared-model-GGUF"}
    likes_only_model = {"id": "org/likes-only-model-GGUF"}

    async def fake_fetch(session: Mock, sort: str, limit: int) -> list[dict[str, str]]:
        if sort == "downloads":
            return [shared_model]
        return [shared_model, likes_only_model]

    with patch("scripts.get_llamacpp_models.fetch_popular_gguf_models", side_effect=fake_fetch):
        result = await collect_gguf_models(MagicMock(), {"downloads": 10, "likes": 10})

    ids = [m["id"] for m in result]
    assert ids == ["org/shared-model-GGUF", "org/likes-only-model-GGUF"]


@pytest.mark.asyncio
async def test_fetch_model_files_returns_siblings() -> None:
    data = {"siblings": [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]}
    session = _make_mock_session(data)
    sem = asyncio.Semaphore(1)

    siblings = await fetch_model_files(session, "org/model", sem)

    assert siblings == data["siblings"]


@pytest.mark.asyncio
async def test_fetch_model_files_returns_none_on_error(capsys: pytest.CaptureFixture[str]) -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=404))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    session = MagicMock()
    session.get = MagicMock(return_value=mock_resp)
    sem = asyncio.Semaphore(1)

    siblings = await fetch_model_files(session, "org/missing", sem)

    assert siblings is None
    assert "org/missing" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_fetch_llamacpp_entries_raises_when_no_sort_active() -> None:
    with pytest.raises(ValueError, match="Specify at least one"):
        await fetch_llamacpp_entries(MagicMock(), 0, 0, 0, log=lambda _msg: None)


@pytest.mark.asyncio
async def test_main_no_active_sort_prints_error(capsys: pytest.CaptureFixture[str]) -> None:
    await main(0, 0, 0, raw=False)

    captured = capsys.readouterr()
    assert "Specify at least one" in captured.err


@pytest.mark.asyncio
async def test_main_emits_multiple_entries_per_repo(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/model-GGUF"}]
    files = [[{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}, {"rfilename": "model-Q8_0.gguf", "size": 2 * 1024**3}]]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return files[0]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True)

    out = capsys.readouterr().out
    data = json.loads(out)
    names = [e["name"] for e in data["llms"]]
    assert names == ["org/model-GGUF-q4-k-m", "org/model-GGUF-q8-0"]


@pytest.mark.asyncio
async def test_main_sorts_entries_case_insensitively_by_name(capsys: pytest.CaptureFixture[str]) -> None:
    # Repos are returned out of alphabetical order, and mixed case, to prove `main` re-sorts them
    # rather than preserving fetch order.
    models = [{"id": "org/Zebra-GGUF"}, {"id": "org/apple-GGUF"}, {"id": "org/Mango-GGUF"}]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True)

    out = capsys.readouterr().out
    data = json.loads(out)
    names = [e["name"] for e in data["llms"]]
    assert names == ["org/apple-GGUF-q4-k-m", "org/Mango-GGUF-q4-k-m", "org/Zebra-GGUF-q4-k-m"]


@pytest.mark.asyncio
async def test_main_drops_repo_without_gguf_files(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": "org/model-GGUF"}]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return [{"rfilename": "config.json", "size": 100}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False)

    out = capsys.readouterr()
    data = json.loads(out.out)
    assert data["llms"] == []
    assert "Dropped 1 repo" in out.err


@pytest.mark.asyncio
async def test_main_distinguishes_failed_fetch_from_empty_repo(capsys: pytest.CaptureFixture[str]) -> None:
    # 20 repos so 1 failure (5%) stays under the 10% abort threshold.
    models = [{"id": f"org/model-ok-{i}"} for i in range(18)] + [{"id": "org/model-empty"}, {"id": "org/model-failed"}]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(
        session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None
    ) -> list[dict[str, Any]] | None:
        if model_id == "org/model-empty":
            return [{"rfilename": "config.json", "size": 100}]
        if model_id == "org/model-failed":
            return None
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=False)

    out = capsys.readouterr()
    data = json.loads(out.out)
    assert len(data["llms"]) == 18
    assert "Failed to fetch file listing for 1 repo" in out.err
    assert "Dropped 1 repo" in out.err


@pytest.mark.asyncio
async def test_main_aborts_when_fetch_failure_rate_exceeds_threshold(capsys: pytest.CaptureFixture[str]) -> None:
    models = [{"id": f"org/model-{i}"} for i in range(10)]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(
        session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None
    ) -> list[dict[str, Any]] | None:
        # 3/10 = 30% failure, above the 10% threshold.
        if model_id in ("org/model-0", "org/model-1", "org/model-2"):
            return None
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        await main(10, 0, 0, raw=False)

    assert exc_info.value.code == 1
    assert "exceeds" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_main_output_writes_file_and_preserves_other_key(tmp_path: Path) -> None:
    output_path = tmp_path / "llamacpp-min.json"
    output_path.write_text(json.dumps({"other_key": ["unrelated"]}))

    models = [{"id": "org/model-GGUF"}]

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return models

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, output=str(output_path))

    data = json.loads(output_path.read_text())
    assert data["llms"] == [
        {
            "name": "org/model-GGUF-q4-k-m",
            "url": "https://huggingface.co/org/model-GGUF/resolve/main/model-Q4_K_M.gguf",
            "size": "1.1GB",
            "jinja": False,
            "supports_vision": None,
            "has_mmproj": None,
        }
    ]
    assert data["other_key"] == ["unrelated"]


@pytest.mark.asyncio
async def test_main_output_preserves_jinja_flag_for_reappearing_entry(tmp_path: Path) -> None:
    output_path = tmp_path / "llamacpp-min.json"
    output_path.write_text(
        json.dumps(
            {
                "llms": [
                    {
                        "name": "org/legacy-curated-name",
                        "url": "https://huggingface.co/org/model-GGUF/resolve/main/model-Q4_K_M.gguf",
                        "size": "1GB",
                        "jinja": True,
                    }
                ]
            }
        )
    )

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return [{"id": "org/model-GGUF"}]

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, output=str(output_path))

    data = json.loads(output_path.read_text())
    assert data["llms"] == [
        {
            "name": "org/model-GGUF-q4-k-m",
            "url": "https://huggingface.co/org/model-GGUF/resolve/main/model-Q4_K_M.gguf",
            "size": "1.1GB",
            "jinja": True,
            "supports_vision": None,
            "has_mmproj": None,
        }
    ]


@pytest.mark.asyncio
async def test_main_output_overwrites_llms_key(tmp_path: Path) -> None:
    output_path = tmp_path / "llamacpp-min.json"
    output_path.write_text(json.dumps({"llms": [{"name": "org/old-model", "url": "https://old", "size": "1GB"}]}))

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return [{"id": "org/new-model-GGUF"}]

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        return [{"rfilename": "new-model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(10, 0, 0, raw=True, output=str(output_path))

    data = json.loads(output_path.read_text())
    names = [e["name"] for e in data["llms"]]
    assert names == ["org/new-model-GGUF-q4-k-m"]


@pytest.mark.asyncio
async def test_main_passes_default_cache_dir_to_fetch_model_files() -> None:
    seen: list[Path | None] = []

    async def fake_collect(session: Mock, active: dict[str, int]) -> list[dict[str, str]]:
        return [{"id": "org/model-GGUF"}]

    async def fake_files(session: Mock, model_id: str, sem: asyncio.Semaphore, cache_dir: Path | None = None) -> list[dict[str, Any]]:
        seen.append(cache_dir)
        return [{"rfilename": "model-Q4_K_M.gguf", "size": 1024**3}]

    with (
        patch("scripts.get_llamacpp_models.collect_gguf_models", side_effect=fake_collect),
        patch("scripts.get_llamacpp_models.fetch_model_files", side_effect=fake_files),
        patch(
            "aiohttp.ClientSession",
            return_value=AsyncMock(__aenter__=AsyncMock(return_value=MagicMock()), __aexit__=AsyncMock(return_value=False)),
        ),
    ):
        await main(1, 0, 0, raw=True)

    assert seen == [DEFAULT_MODEL_DETAIL_CACHE_DIR]
