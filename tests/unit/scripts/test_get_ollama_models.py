# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for get_ollama_models.py script."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest

from scripts.get_ollama_models import (
    Category,
    ExtraSpec,
    MinJson,
    ModelEntry,
    ModelInfo,
    build_min_json,
    capability_to_category,
    deduplicate_by_hash,
    extract_size,
    fetch,
    fetch_cached,
    fetch_extra_model,
    fetch_model_tags,
    fetch_with_fallback,
    fill_sizes_from_manifests,
    find_all_between,
    format_size,
    guess_category,
    main,
    manifest_url,
    parse_context,
    parse_library,
    parse_model_page,
    parse_tags_full,
    parse_tags_hashes,
    size_from_manifest,
    skip_suffixes,
)


@pytest.mark.parametrize(
    ("html", "attr", "open_delim", "close_delim", "expected"),
    [
        pytest.param(
            '<span x-test-capability class="">embedding</span>',
            "x-test-capability",
            ">",
            "<",
            ["embedding"],
            id="single_match",
        ),
        pytest.param(
            '<span x-test-capability class="">embed</span> <span x-test-capability class="">vision</span>',
            "x-test-capability",
            ">",
            "<",
            ["embed", "vision"],
            id="multiple_matches",
        ),
        pytest.param(
            "no attr here",
            "x-test-capability",
            ">",
            "<",
            [],
            id="no_match",
        ),
        pytest.param(
            "x-test-capability",
            "x-test-capability",
            ">",
            "EOF",
            [],
            id="attr_present_but_no_open_tag",
        ),
        pytest.param(
            '<span x-test-capability class="">  value  </span>',
            "x-test-capability",
            ">",
            "<",
            ["value"],
            id="strips_whitespace",
        ),
        pytest.param(
            '<span x-test-capability class="">no closing tag ever',
            "x-test-capability",
            ">",
            "</span>",
            [],
            id="close_tag_not_found",
        ),
    ],
)
def test_find_all_between(html: str, attr: str, open_delim: str, close_delim: str, expected: list[str]) -> None:
    assert find_all_between(html, attr, open_delim, close_delim) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("2048", 2048),
        ("128K", 128 * 1024),
        ("1M", 1024 * 1024),
        ("0.5K", 512),
        ("invalid", None),
        ("  64K  ", 64 * 1024),
    ],
)
def test_parse_context(value: str | None, expected: int | None) -> None:
    assert parse_context(value) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("llama3:cloud", True),
        ("some-model-cloud", True),
        ("llama3:8b", False),
        ("mistral", False),
    ],
)
def test_skip_suffixes(name: str, expected: bool) -> None:
    assert skip_suffixes(name) is expected


def _entry(name: str, hash_: str) -> ModelEntry:
    return {"name": name, "size": "", "hash": hash_, "context": None}


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([_entry("a", ""), _entry("b", "")], id="no_hash_always_kept"),
        pytest.param([_entry("a:t1", "h1"), _entry("a:t2", "h2")], id="different_hashes_all_kept"),
    ],
)
def test_deduplicate_by_hash_all_kept(entries: list[ModelEntry]) -> None:
    result = deduplicate_by_hash(entries)

    assert len(result) == 2


def test_deduplicate_by_hash_colon_and_no_colon_same_hash_both_kept() -> None:
    result = deduplicate_by_hash([_entry("model", "h1"), _entry("model:tag", "h1")])

    names = {e["name"] for e in result}
    assert names == {"model", "model:tag"}


@pytest.mark.parametrize(
    ("entries", "kept", "removed"),
    [
        ([_entry("model:long-tag", "abc"), _entry("model:x", "abc")], "model:x", "model:long-tag"),
        ([_entry("a-longer-name", "xyz"), _entry("short", "xyz")], "short", "a-longer-name"),
        ([_entry("short", "xyz"), _entry("much-longer-name", "xyz")], "short", "much-longer-name"),
        ([_entry("m:x", "h1"), _entry("m:long-tag-name", "h1")], "m:x", "m:long-tag-name"),
    ],
    ids=["colon_prefers_shorter", "no_colon_prefers_shorter", "order_invariant", "colon_long_dropped"],
)
def test_deduplicate_by_hash_prefers_shorter(entries: list[ModelEntry], kept: str, removed: str) -> None:
    names = [e["name"] for e in deduplicate_by_hash(entries)]

    assert kept in names
    assert removed not in names


@pytest.mark.parametrize(
    ("caps", "expected"),
    [
        (["Embedding"], "embeddings"),
        (["image generation"], "txt2img"),
        (["text-to-image"], "txt2img"),
        (["txt2img"], "txt2img"),
        (["vision", "tools"], "llms"),
        ([], "llms"),
        (["tools", "decision", "thinking"], "decision"),
        (["Decision"], "decision"),
    ],
    ids=["embedding", "image_generation", "text_to_image", "txt2img", "vision_tools_llms", "empty_llms", "decision_tools", "decision_only"],
)
def test_capability_to_category(caps: list[str], expected: str) -> None:
    assert capability_to_category(caps) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nomic-embed-text", "embeddings"),
        ("all-MiniLM-L6-v2", "embeddings"),
        ("e5-small", "embeddings"),
        ("flux-dev", "txt2img"),
        ("stable-diffusion", "txt2img"),
        ("llama3:8b", "llms"),
    ],
    ids=["embed_keyword", "minilm_keyword", "e5_keyword", "flux_keyword", "diffusion_keyword", "llm_default"],
)
def test_guess_category(name: str, expected: str) -> None:
    assert guess_category(name) == expected


_LIBRARY_HTML = """
<ul>
  <li  class="flex items-baseline border-b border-neutral-200 py-6">
    <a href="/library/llama3" class="group w-full space-y-5">
      <span class="group-hover:underline truncate">llama3</span>
      <span class="text-indigo-600">tools</span>
    </a>
  </li>
  <li  class="flex items-baseline border-b border-neutral-200 py-6">
    <a href="/llama-no-lib" class="group w-full space-y-5">skip</a>
  </li>
  <li no-attr>ignored</li>
</ul>
"""


def test_parse_library_parses_model_name() -> None:
    assert "llama3" in [m["name"] for m in parse_library(_LIBRARY_HTML)]


def test_parse_library_parses_capabilities() -> None:
    llama = next(m for m in parse_library(_LIBRARY_HTML) if m["name"] == "llama3")

    assert "tools" in llama["capabilities"]


def test_parse_library_ignores_li_without_expected_class() -> None:
    assert "ignored" not in [m["name"] for m in parse_library(_LIBRARY_HTML)]


def test_parse_library_empty_html() -> None:
    assert parse_library("") == []


def test_parse_library_li_with_expected_class_but_no_href() -> None:
    html = '<li  class="flex items-baseline border-b border-neutral-200 py-6"><span>no href here</span></li>'

    assert parse_library(html) == []


_TAGS_HTML = """
<div class="group px-4 py-3">
  <a href="/library/llama3:8b">llama3:8b</a>
  <span class="font-mono text-xs">abc123  </span>
</div>
<div class="group px-4 py-3">
  <a href="/library/llama3:latest">llama3:latest</a>
  <span class="font-mono text-xs">def456</span>
</div>
"""


def test_parse_tags_hashes_basic() -> None:
    assert parse_tags_hashes(_TAGS_HTML).get("llama3:8b") == "abc123"


def test_parse_tags_hashes_strips_latest() -> None:
    assert "llama3" in parse_tags_hashes(_TAGS_HTML)


@pytest.mark.parametrize(
    "html",
    [
        "no parts",
        '<div class="group px-4 py-3">no href and no span here</div>',
    ],
    ids=["no_parts", "no_href_or_hash"],
)
def test_parse_tags_hashes_returns_empty(html: str) -> None:
    assert parse_tags_hashes(html) == {}


def test_parse_tags_full_parses_entries() -> None:
    assert "llama3:8b" in [e["name"] for e in parse_tags_full(_TAGS_HTML)]


def test_parse_tags_full_hash_populated() -> None:
    entry = next(e for e in parse_tags_full(_TAGS_HTML) if e["name"] == "llama3:8b")

    assert entry["hash"] == "abc123"


def test_parse_tags_full_size_rejects_usage_slot_widget_markup() -> None:
    html = """
    <div class="group px-4 py-3">
      <a href="/library/cloudmodel:preview">cloudmodel:preview</a>
      <p class="col-span-2">
            <span x-test-model-tag-usage-slot-active class="block h-1 w-4 rounded-full bg-neutral-800"></span>
      </p>
    </div>
    """
    entry = next(e for e in parse_tags_full(html) if e["name"] == "cloudmodel:preview")

    assert entry["size"] == ""


@pytest.mark.parametrize(
    "html",
    [
        "no parts",
        '<div class="group px-4 py-3">no href here at all</div>',
    ],
    ids=["no_parts", "no_href"],
)
def test_parse_tags_full_returns_empty(html: str) -> None:
    assert parse_tags_full(html) == []


_MODEL_HTML = """
<div>
  <span x-test-model-namespace class="">myorg</span>
</div>
<div class="group px-4 py-3">
  <a class="block something">mymodel:7b</a>
  <p class="col-span-2">4.1 GB</p>
  <p class="col-span-2">128K</p>
</div>
<div class="group px-4 py-3">
  <a class="block something">mymodel:cloud</a>
  <p class="col-span-2">4.1 GB</p>
</div>
"""


def test_parse_model_page_basic_entry() -> None:
    assert any("mymodel:7b" in e["name"] for e in parse_model_page(_MODEL_HTML, None))


def test_parse_model_page_skip_suffixes_applied() -> None:
    assert not any("cloud" in e["name"] for e in parse_model_page(_MODEL_HTML, None))


def test_parse_model_page_size_parsed() -> None:
    entry = next(e for e in parse_model_page(_MODEL_HTML, None) if "7b" in e["name"])

    assert entry["size"] == "4.1 GB"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("4.1 GB", "4.1 GB", id="plain"),
        pytest.param("292MB", "292MB", id="no_space"),
        pytest.param("  8.6GB  ", "8.6GB", id="whitespace"),
        pytest.param(
            '<span x-test-model-tag-usage-slot-active class="block h-1 w-4 rounded-full bg-neutral-800"></span>',
            "",
            id="usage_slot_widget_markup",
        ),
        pytest.param("", "", id="empty"),
        pytest.param("not a size", "", id="garbage_text"),
    ],
)
def test_extract_size(raw: str, expected: str) -> None:
    assert extract_size(raw) == expected


def test_extract_size_rejects_size_range() -> None:
    assert extract_size("9.3GB - 9.5GB") == ""


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        pytest.param(9_527_504_063, "9.5GB", id="one_decimal_below_ten"),
        pytest.param(5_000_000_000, "5.0GB", id="keeps_trailing_zero"),
        pytest.param(833_606_519, "834MB", id="whole_number_from_ten"),
        pytest.param(18_000_000_000, "18GB", id="whole_gigabytes"),
        pytest.param(999_600_000, "1.0GB", id="rounds_up_to_next_unit"),
        pytest.param(9_960_000_000, "10GB", id="rounds_up_to_ten"),
        pytest.param(512, "512B", id="bytes"),
        pytest.param(2_500_000_000_000_000, "2500TB", id="largest_unit"),
    ],
)
def test_format_size_matches_ollama_com_format(size: int, expected: str) -> None:
    assert format_size(size) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("nimble", "https://registry.ollama.ai/v2/library/nimble/manifests/latest", id="no_tag"),
        pytest.param("nimble:9b", "https://registry.ollama.ai/v2/library/nimble/manifests/9b", id="tag"),
        pytest.param("myorg/mymodel:7b", "https://registry.ollama.ai/v2/myorg/mymodel/manifests/7b", id="namespace"),
    ],
)
def test_manifest_url_builds_registry_path(name: str, expected: str) -> None:
    assert manifest_url(name) == expected


def _manifest(*layers: tuple[str, int]) -> str:
    return json.dumps({"layers": [{"mediaType": f"application/vnd.ollama.image.{kind}", "size": size} for kind, size in layers]})


def test_size_from_manifest_sums_all_layers() -> None:
    assert size_from_manifest(_manifest(("model", 9_527_000_000), ("license", 400_000), ("params", 100_000))) == "9.5GB"


@pytest.mark.parametrize(
    "manifest",
    [
        pytest.param(_manifest(("license", 400), ("params", 100)), id="no_model_layer"),
        pytest.param(json.dumps({"schemaVersion": 2}), id="no_layers"),
    ],
)
def test_size_from_manifest_without_model_layer_returns_empty(manifest: str) -> None:
    assert size_from_manifest(manifest) == ""


def test_parse_model_page_size_rejects_usage_slot_widget_markup() -> None:
    html = """
    <div class="group px-4 py-3">
      <a class="block something">cloudmodel:preview</a>
      <p class="col-span-2">
            <span x-test-model-tag-usage-slot-active class="block h-1 w-4 rounded-full bg-neutral-800"></span>
      </p>
    </div>
    """
    entry = next(e for e in parse_model_page(html, None) if "preview" in e["name"])

    assert entry["size"] == ""


def test_parse_model_page_context_parsed() -> None:
    entry = next(e for e in parse_model_page(_MODEL_HTML, None) if "7b" in e["name"])

    assert entry["context"] == 128 * 1024


def test_parse_model_page_only_tags_filter() -> None:
    assert parse_model_page(_MODEL_HTML, ["nonexistent"]) == []


def test_parse_model_page_namespace_prepended() -> None:
    assert all(e["name"].startswith("myorg/") for e in parse_model_page(_MODEL_HTML, None))


def test_parse_model_page_part_without_block_class_skipped() -> None:
    html = '<div class="group px-4 py-3"><a class="other-class">model:7b</a></div>'

    assert parse_model_page(html, None) == []


@pytest.mark.asyncio
async def test_fetch_returns_text() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = AsyncMock(return_value="hello")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    assert await fetch(mock_session, "http://example.com") == "hello"


@pytest.mark.asyncio
async def test_fetch_raises_on_error() -> None:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(MagicMock(), (), status=500))
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    with pytest.raises(aiohttp.ClientResponseError):
        await fetch(mock_session, "http://example.com")


@pytest.mark.asyncio
async def test_fetch_cached_reads_from_cache_when_exists(tmp_path: Path) -> None:
    cache_file = tmp_path / "cached.html"
    cache_file.write_text("cached content", encoding="utf-8")
    mock_session = MagicMock()

    result = await fetch_cached(mock_session, "http://example.com", cache_file, use_cache=True)

    assert result == "cached content"
    assert mock_session.get.call_count == 0


@pytest.mark.asyncio
async def test_fetch_cached_fetches_and_writes_when_no_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "sub" / "new.html"
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = AsyncMock(return_value="fresh")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    result = await fetch_cached(mock_session, "http://example.com", cache_file, use_cache=False)

    assert result == "fresh"
    assert cache_file.read_text(encoding="utf-8") == "fresh"


@pytest.mark.asyncio
async def test_fetch_cached_fetches_when_cache_disabled_but_file_exists(tmp_path: Path) -> None:
    cache_file = tmp_path / "cached.html"
    cache_file.write_text("old", encoding="utf-8")
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.text = AsyncMock(return_value="fresh")
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)

    result = await fetch_cached(mock_session, "http://example.com", cache_file, use_cache=False)

    assert result == "fresh"


@pytest.mark.asyncio
async def test_fetch_with_fallback_returns_from_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / "c.html"
    cache_file.write_text("cached", encoding="utf-8")

    result = await fetch_with_fallback(MagicMock(), ["http://a.com"], cache_file, use_cache=True)

    assert result == "cached"


@pytest.mark.asyncio
async def test_fetch_with_fallback_falls_back_on_404(tmp_path: Path) -> None:
    cache_file = tmp_path / "c.html"
    exc_404 = aiohttp.ClientResponseError(MagicMock(), (), status=404)
    call_count = 0

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise exc_404
        return "second"

    with patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached):
        result = await fetch_with_fallback(MagicMock(), ["http://first.com", "http://second.com"], cache_file, use_cache=False)

    assert result == "second"


@pytest.mark.asyncio
async def test_fetch_with_fallback_raises_non_404_immediately(tmp_path: Path) -> None:
    cache_file = tmp_path / "c.html"
    exc_500 = aiohttp.ClientResponseError(MagicMock(), (), status=500)

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        raise exc_500

    with (
        patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached),
        pytest.raises(aiohttp.ClientResponseError) as exc_info,
    ):
        await fetch_with_fallback(MagicMock(), ["http://first.com", "http://second.com"], cache_file, use_cache=False)

    assert exc_info.value.status == 500


@pytest.mark.asyncio
async def test_fetch_with_fallback_raises_404_when_all_candidates_exhausted(tmp_path: Path) -> None:
    cache_file = tmp_path / "c.html"
    exc_404 = aiohttp.ClientResponseError(MagicMock(), (), status=404)

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        raise exc_404

    with patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached), pytest.raises(aiohttp.ClientResponseError):
        await fetch_with_fallback(MagicMock(), ["http://only.com"], cache_file, use_cache=False)


@pytest.mark.asyncio
async def test_fetch_with_fallback_empty_candidates_raises(tmp_path: Path) -> None:
    cache_file = tmp_path / "c.html"
    with pytest.raises((TypeError, Exception)):
        await fetch_with_fallback(MagicMock(), [], cache_file, use_cache=False)


_SIMPLE_MODEL_HTML = """\
<div class="group px-4 py-3">
  <a class="block something">testmodel:7b</a>
  <p class="col-span-2">4 GB</p>
  <p class="col-span-2">128K</p>
</div>
"""

_SIMPLE_TAGS_HTML = """\
<div class="group px-4 py-3">
  <a href="/library/testmodel:7b">testmodel:7b</a>
  <span class="font-mono text-xs">abc123</span>
</div>
"""


def _fake_fetch_cached_tags_and_model(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
    return _SIMPLE_TAGS_HTML if "tags" in url else _SIMPLE_MODEL_HTML


def _entry_with_size(name: str, size: str) -> ModelEntry:
    return {"name": name, "size": size, "hash": "", "context": None}


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
async def test_fill_sizes_from_manifests_fills_empty_size(mock_fetch_cached: AsyncMock) -> None:
    mock_fetch_cached.return_value = _manifest(("model", 9_527_000_000))
    _mock_session = MagicMock()
    entries = [_entry_with_size("nimble:9b", "")]

    result = await fill_sizes_from_manifests(_mock_session, asyncio.Semaphore(1), entries, True)

    assert result == [_entry_with_size("nimble:9b", "9.5GB")]
    assert mock_fetch_cached.call_count == 1
    assert mock_fetch_cached.call_args == call(
        _mock_session,
        "https://registry.ollama.ai/v2/library/nimble/manifests/9b",
        Path("./ollama/manifests/nimble_9b.json"),
        True,
    )


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
async def test_fill_sizes_from_manifests_keeps_known_size(mock_fetch_cached: AsyncMock) -> None:
    entries = [_entry_with_size("nimble:9b-q8_0", "9.5GB")]

    result = await fill_sizes_from_manifests(MagicMock(), asyncio.Semaphore(1), entries, False)

    assert result == [_entry_with_size("nimble:9b-q8_0", "9.5GB")]
    assert mock_fetch_cached.call_count == 0


@pytest.mark.parametrize(
    "fetch_result",
    [
        pytest.param(OSError("network error"), id="fetch_failure"),
        pytest.param(["not a manifest"], id="invalid_json"),
    ],
)
@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
async def test_fill_sizes_from_manifests_leaves_size_empty_on_error(
    mock_fetch_cached: AsyncMock, fetch_result: Exception | list[str]
) -> None:
    mock_fetch_cached.side_effect = fetch_result
    entries = [_entry_with_size("nimble:9b", "")]

    result = await fill_sizes_from_manifests(MagicMock(), asyncio.Semaphore(1), entries, False)

    assert result == [_entry_with_size("nimble:9b", "")]
    assert mock_fetch_cached.call_count == 1


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
async def test_fetch_model_tags_fills_size_range_from_manifest(mock_fetch_cached: AsyncMock) -> None:
    model_html = _SIMPLE_MODEL_HTML.replace("4 GB", "9.3GB - 9.5GB")
    mock_fetch_cached.side_effect = (model_html, _SIMPLE_TAGS_HTML, _manifest(("model", 9_527_000_000)))
    _mock_session = MagicMock()
    model: ModelInfo = {"name": "testmodel", "capabilities": []}

    _, _, entries = await fetch_model_tags(_mock_session, asyncio.Semaphore(1), model, False, {})

    assert [(e["name"], e["size"]) for e in entries] == [("testmodel:7b", "9.5GB")]
    assert mock_fetch_cached.call_count == 3
    assert mock_fetch_cached.call_args == call(
        _mock_session,
        "https://registry.ollama.ai/v2/library/testmodel/manifests/7b",
        Path("./ollama/manifests/testmodel_7b.json"),
        False,
    )


@pytest.mark.asyncio
async def test_fetch_model_tags_successful_fetch_returns_entries() -> None:
    model: ModelInfo = {"name": "testmodel", "capabilities": []}

    with patch(
        "scripts.get_ollama_models.fetch_cached",
        side_effect=_fake_fetch_cached_tags_and_model,
    ):
        category, name, entries = await fetch_model_tags(MagicMock(), asyncio.Semaphore(1), model, False, {})

    assert name == "testmodel"
    assert category == "llms"
    assert any("testmodel:7b" in e["name"] for e in entries)


@pytest.mark.asyncio
async def test_fetch_model_tags_hash_merged_from_tags_page() -> None:
    model: ModelInfo = {"name": "testmodel", "capabilities": []}

    with patch(
        "scripts.get_ollama_models.fetch_cached",
        side_effect=_fake_fetch_cached_tags_and_model,
    ):
        _, _, entries = await fetch_model_tags(MagicMock(), asyncio.Semaphore(1), model, False, {})

    entry = next(e for e in entries if "testmodel:7b" in e["name"])
    assert entry["hash"] == "abc123"


@pytest.mark.asyncio
async def test_fetch_model_tags_fetch_failure_returns_empty() -> None:
    model: ModelInfo = {"name": "badmodel", "capabilities": []}

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        raise OSError("network error")

    with patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached):
        _category, name, entries = await fetch_model_tags(MagicMock(), asyncio.Semaphore(1), model, False, {})

    assert name == "badmodel"
    assert entries == []


@pytest.mark.asyncio
async def test_fetch_model_tags_pick_tags_filters_entries() -> None:
    model: ModelInfo = {"name": "testmodel", "capabilities": []}

    with patch(
        "scripts.get_ollama_models.fetch_cached",
        side_effect=_fake_fetch_cached_tags_and_model,
    ):
        _, _, entries = await fetch_model_tags(MagicMock(), asyncio.Semaphore(1), model, False, {"testmodel": ["nonexistent"]})

    assert entries == []


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
@patch("scripts.get_ollama_models.fetch_with_fallback", new_callable=AsyncMock)
async def test_fetch_extra_model_specific_tag_returns_matched_entry(
    mock_fetch_with_fallback: AsyncMock, mock_fetch_cached: AsyncMock
) -> None:
    mock_fetch_with_fallback.return_value = _TAGS_HTML
    mock_fetch_cached.return_value = _manifest(("model", 4_700_000_000))

    category, name, entries = await fetch_extra_model(MagicMock(), asyncio.Semaphore(1), "llama3:8b", "llms", False, {})

    assert name == "llama3"
    assert category == "llms"
    assert [(e["name"], e["size"]) for e in entries] == [("llama3:8b", "4.7GB")]


@pytest.mark.asyncio
async def test_fetch_extra_model_specific_tag_no_match_returns_empty() -> None:
    async def fake_fallback(session: aiohttp.ClientSession, candidates: list[str], path: Path, use_cache: bool) -> str:
        return _TAGS_HTML

    with patch("scripts.get_ollama_models.fetch_with_fallback", side_effect=fake_fallback):
        _, _, entries = await fetch_extra_model(MagicMock(), asyncio.Semaphore(1), "llama3:nonexistent", "llms", False, {})

    assert entries == []


@pytest.mark.asyncio
async def test_fetch_extra_model_no_tag_fetches_model_and_tags_pages() -> None:
    call_count = 0

    async def fake_fallback(session: aiohttp.ClientSession, candidates: list[str], path: Path, use_cache: bool) -> str:
        nonlocal call_count
        call_count += 1
        return _SIMPLE_TAGS_HTML if call_count == 1 else _SIMPLE_MODEL_HTML

    with patch("scripts.get_ollama_models.fetch_with_fallback", side_effect=fake_fallback):
        category, name, _entries = await fetch_extra_model(MagicMock(), asyncio.Semaphore(1), "testmodel", "embeddings", False, {})

    assert name == "testmodel"
    assert category == "embeddings"
    assert call_count == 2


@pytest.mark.asyncio
async def test_fetch_extra_model_tags_fetch_failure_returns_empty() -> None:
    async def fake_fallback(session: aiohttp.ClientSession, candidates: list[str], path: Path, use_cache: bool) -> str:
        raise OSError("network error")

    with patch("scripts.get_ollama_models.fetch_with_fallback", side_effect=fake_fallback):
        _category, name, entries = await fetch_extra_model(MagicMock(), asyncio.Semaphore(1), "mymodel", "embeddings", False, {})

    assert name == "mymodel"
    assert entries == []


@pytest.mark.asyncio
async def test_fetch_extra_model_model_page_fetch_failure_returns_empty() -> None:
    call_count = 0

    async def fake_fallback(session: aiohttp.ClientSession, candidates: list[str], path: Path, use_cache: bool) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SIMPLE_TAGS_HTML
        raise OSError("model page failed")

    with patch("scripts.get_ollama_models.fetch_with_fallback", side_effect=fake_fallback):
        _, _, entries = await fetch_extra_model(MagicMock(), asyncio.Semaphore(1), "mymodel", "llms", False, {})

    assert entries == []


async def _fake_fetch_model_tags_llms(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    model: ModelInfo,
    use_cache: bool,
    pick_tags: dict[str, list[str]],
) -> tuple[Category, str, list[ModelEntry]]:
    return ("llms", model["name"], [])


@pytest.mark.asyncio
async def test_build_min_json_returns_categorized_output() -> None:
    extra_specs: ExtraSpec = {
        "extra": {"embeddings": [], "llms": [], "txt2img": []},
        "pickTags": {},
    }

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        return _LIBRARY_HTML

    with (
        patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached),
        patch("scripts.get_ollama_models.fetch_model_tags", side_effect=_fake_fetch_model_tags_llms),
    ):
        result = await build_min_json(use_cache=False, concurrency=2, extra_specs=extra_specs)

    assert set(result.keys()) == {"embeddings", "llms", "txt2img", "decision"}


@pytest.mark.asyncio
async def test_build_min_json_extra_specs_models_included() -> None:
    extra_specs: ExtraSpec = {
        "extra": {"embeddings": ["nomic-embed-text"], "llms": [], "txt2img": []},
        "pickTags": {},
    }

    async def fake_fetch_cached(session: aiohttp.ClientSession, url: str, path: Path, use_cache: bool) -> str:
        return _LIBRARY_HTML

    async def fake_fetch_extra(
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        spec: str,
        category: Category,
        use_cache: bool,
        pick_tags: dict[str, list[str]],
    ) -> tuple[Category, str, list[ModelEntry]]:
        return (
            category,
            spec,
            [{"name": spec, "size": "500 MB", "hash": "abc", "context": None}],
        )

    with (
        patch("scripts.get_ollama_models.fetch_cached", side_effect=fake_fetch_cached),
        patch("scripts.get_ollama_models.fetch_model_tags", side_effect=_fake_fetch_model_tags_llms),
        patch("scripts.get_ollama_models.fetch_extra_model", side_effect=fake_fetch_extra),
    ):
        result = await build_min_json(use_cache=False, concurrency=2, extra_specs=extra_specs)

    assert any(e["name"] == "nomic-embed-text" for e in result["embeddings"])


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.fetch_model_tags", new_callable=AsyncMock)
@patch("scripts.get_ollama_models.fetch_cached", new_callable=AsyncMock)
async def test_build_min_json_puts_decision_models_in_decision_category(
    mock_fetch_cached: AsyncMock, mock_fetch_model_tags: AsyncMock
) -> None:
    extra_specs: ExtraSpec = {"extra": {"embeddings": [], "llms": [], "txt2img": []}, "pickTags": {}}
    mock_fetch_cached.return_value = _LIBRARY_HTML
    mock_fetch_model_tags.return_value = ("decision", "tev1", [{"name": "tev1", "size": "", "hash": "h", "context": None}])

    result = await build_min_json(use_cache=False, concurrency=2, extra_specs=extra_specs)

    assert result["decision"]
    assert result["llms"] == []


@pytest.mark.asyncio
async def test_main_clear_cache_existing_dir(tmp_path: Path) -> None:
    cache = tmp_path / "ollama"
    cache.mkdir()

    with (
        patch("sys.argv", ["script", "--clear-cache"]),
        patch("scripts.get_ollama_models.CACHE_DIR", cache),
    ):
        await main()

    assert not cache.exists()


@pytest.mark.asyncio
async def test_main_clear_cache_nonexistent_dir(tmp_path: Path) -> None:
    cache = tmp_path / "ollama_gone"

    with (
        patch("sys.argv", ["script", "--clear-cache"]),
        patch("scripts.get_ollama_models.CACHE_DIR", cache),
    ):
        await main()


@pytest.mark.asyncio
async def test_main_normal_run_writes_output(tmp_path: Path) -> None:
    output = tmp_path / "out.json"

    async def fake_build(use_cache: bool, concurrency: int, extra_specs: ExtraSpec) -> MinJson:
        return {"embeddings": [], "llms": [], "txt2img": [], "decision": []}

    with (
        patch(
            "sys.argv",
            ["script", "--output", str(output), "--extra", str(tmp_path / "no_extra.json"), "--merge", str(tmp_path / "no_merge.json")],
        ),
        patch("scripts.get_ollama_models.build_min_json", side_effect=fake_build),
    ):
        await main()

    assert output.exists()
    assert set(json.loads(output.read_text()).keys()) == {"embeddings", "llms", "txt2img", "decision"}


@pytest.mark.asyncio
async def test_main_extra_file_loaded(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    extra_file = tmp_path / "extra.json"
    extra_specs: ExtraSpec = {
        "extra": {"embeddings": [], "llms": ["llama3"], "txt2img": []},
        "pickTags": {},
    }
    extra_file.write_text(json.dumps(extra_specs), encoding="utf-8")
    captured: list[ExtraSpec] = []

    async def fake_build(use_cache: bool, concurrency: int, extra_specs: ExtraSpec) -> MinJson:
        captured.append(extra_specs)
        return {"embeddings": [], "llms": [], "txt2img": [], "decision": []}

    with (
        patch(
            "sys.argv",
            ["script", "--output", str(output), "--extra", str(extra_file), "--merge", str(tmp_path / "no_merge.json")],
        ),
        patch("scripts.get_ollama_models.build_min_json", side_effect=fake_build),
    ):
        await main()

    assert captured[0]["extra"]["llms"] == ["llama3"]


@pytest.mark.asyncio
async def test_main_merge_file_appended(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    merge_file = tmp_path / "merge.json"
    merge_file.write_text(
        json.dumps({"llms": [{"name": "custom-model", "size": "1 GB", "hash": "xyz", "context": None}]}),
        encoding="utf-8",
    )

    async def fake_build(use_cache: bool, concurrency: int, extra_specs: ExtraSpec) -> MinJson:
        return {"embeddings": [], "llms": [], "txt2img": [], "decision": []}

    with (
        patch(
            "sys.argv",
            ["script", "--output", str(output), "--extra", str(tmp_path / "no_extra.json"), "--merge", str(merge_file)],
        ),
        patch("scripts.get_ollama_models.build_min_json", side_effect=fake_build),
    ):
        await main()

    data = json.loads(output.read_text())
    assert any(e["name"] == "custom-model" for e in data["llms"])


@pytest.mark.asyncio
@patch("scripts.get_ollama_models.build_min_json", new_callable=AsyncMock)
async def test_main_merge_file_appends_decision_entries(
    mock_build_min_json: AsyncMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.json"
    merge_file = tmp_path / "merge.json"
    merge_file.write_text(
        json.dumps({"decision": [{"name": "my-decider", "size": "1 GB", "hash": "xyz", "context": None}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv", ["script", "--output", str(output), "--extra", str(tmp_path / "no_extra.json"), "--merge", str(merge_file)]
    )
    mock_build_min_json.return_value = {"embeddings": [], "llms": [], "txt2img": [], "decision": []}

    await main()

    assert [e["name"] for e in json.loads(output.read_text())["decision"]] == ["my-decider"]


@pytest.mark.asyncio
async def test_main_output_sorted_by_name(tmp_path: Path) -> None:
    output = tmp_path / "out.json"

    async def fake_build(use_cache: bool, concurrency: int, extra_specs: ExtraSpec) -> MinJson:
        return {
            "embeddings": [],
            "llms": [
                {"name": "zmodel", "size": "", "hash": "", "context": None},
                {"name": "amodel", "size": "", "hash": "", "context": None},
            ],
            "txt2img": [],
            "decision": [],
        }

    with (
        patch(
            "sys.argv",
            ["script", "--output", str(output), "--extra", str(tmp_path / "no_extra.json"), "--merge", str(tmp_path / "no_merge.json")],
        ),
        patch("scripts.get_ollama_models.build_min_json", side_effect=fake_build),
    ):
        await main()

    names = [e["name"] for e in json.loads(output.read_text())["llms"]]
    assert names == sorted(names, key=str.casefold)
