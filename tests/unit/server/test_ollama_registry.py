# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for server/utils/ollama_registry.py."""

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from server.utils.core import DownloadedPacket, DownloadPacket, PreDownloadPacket, SuccessDownloadPacket
from server.utils.ollama_registry import (
    BlobLocks,
    OllamaLayer,
    OllamaModelRef,
    OllamaRegistryClient,
    OllamaRegistryError,
    parse_model_ref,
)

_MODEL_BLOB = b"model-bytes" * 1000
_CONFIG_BLOB = b'{"architecture":"llama"}'
_REF = OllamaModelRef(namespace="library", name="smollm2", tag="135m")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _manifest(model_blob: bytes = _MODEL_BLOB) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "config": {"digest": _digest(_CONFIG_BLOB), "size": len(_CONFIG_BLOB)},
        "layers": [{"digest": _digest(model_blob), "size": len(model_blob)}],
    }


class _FakeRegistry:
    """Serves one model; `corrupt` serves wrong bytes for the model blob, `ignore_range` answers 200 to ranged requests."""

    def __init__(
        self,
        corrupt: bool = False,
        ignore_range: bool = False,
        blob_status: int = 200,
        manifest_status: int = 200,
        reject_range_with_416: bool = False,
        short_body: bool = False,
        redirect_to_cdn: bool = False,
    ) -> None:
        self.corrupt = corrupt
        self.redirect_to_cdn = redirect_to_cdn
        self.gate: asyncio.Event | None = None  # when set, serving the model blob waits for it
        self.reject_range_with_416 = reject_range_with_416
        self.short_body = short_body
        self.ignore_range = ignore_range
        self.blob_status = blob_status
        self.manifest_status = manifest_status
        self.blob_requests: list[tuple[str, str | None]] = []
        self.manifest_text = json.dumps(_manifest())

    async def manifest(self, request: web.Request) -> web.Response:
        if request.match_info["name"] != "smollm2":
            return web.Response(status=404, text="not found")
        return web.Response(status=self.manifest_status, text=self.manifest_text)

    async def blob(self, request: web.Request) -> web.StreamResponse:
        digest = request.match_info["digest"]
        if self.redirect_to_cdn and request.path.startswith("/v2/"):
            location = f"/cdn/{digest}"
            raise web.HTTPTemporaryRedirect(location)
        self.blob_requests.append((digest, request.headers.get("Range")))
        if self.gate is not None and digest == _digest(_MODEL_BLOB):
            await self.gate.wait()
        if self.blob_status != 200:
            return web.Response(status=self.blob_status, text="boom")
        data = {_digest(_CONFIG_BLOB): _CONFIG_BLOB, _digest(_MODEL_BLOB): _MODEL_BLOB}[digest]
        if self.corrupt and digest == _digest(_MODEL_BLOB):
            data = b"x" * len(data)
        range_header = request.headers.get("Range")
        if range_header and self.reject_range_with_416:
            return web.Response(status=416)
        if self.short_body and digest == _digest(_MODEL_BLOB) and not range_header:
            response = web.StreamResponse()
            response.enable_chunked_encoding()
            await response.prepare(request)
            await response.write(data[:100])
            await response.write_eof()
            return response
        if range_header and not self.ignore_range:
            start = int(range_header.removeprefix("bytes=").removesuffix("-"))
            return web.Response(status=206, body=data[start:])
        return web.Response(body=data)


@pytest_asyncio.fixture
async def registry_server(request: pytest.FixtureRequest) -> AsyncGenerator[tuple[_FakeRegistry, TestServer]]:
    registry = _FakeRegistry(**getattr(request, "param", {}))
    app = web.Application()
    app.router.add_get("/v2/{namespace}/{name}/manifests/{tag}", registry.manifest)
    app.router.add_get("/v2/{namespace}/{name}/blobs/{digest}", registry.blob)
    app.router.add_get("/cdn/{digest}", registry.blob)
    server = TestServer(app)
    await server.start_server()
    yield registry, server
    await server.close()


def _client(models_dir: Path, server: TestServer, blob_locks: BlobLocks | None = None) -> OllamaRegistryClient:
    return OllamaRegistryClient(models_dir, registry_url=str(server.make_url("")), blob_locks=blob_locks)


async def _collect(client: OllamaRegistryClient, ref: OllamaModelRef) -> list[DownloadPacket]:
    return [packet async for packet in client.download(ref)]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("llama3.2", ("library", "llama3.2", "latest")),
        ("llama3.2:3b", ("library", "llama3.2", "3b")),
        ("someone/mymodel:v1", ("someone", "mymodel", "v1")),
        ("registry.ollama.ai/library/llama3.2:3b", ("library", "llama3.2", "3b")),
    ],
)
def test_parse_model_ref_parses_default_registry_names(model: str, expected: tuple[str, str, str]) -> None:
    ref = parse_model_ref(model)

    assert ref is not None
    assert (ref.namespace, ref.name, ref.tag) == expected


@pytest.mark.parametrize(
    "model",
    [
        "hf.co/org/repo",
        "hf.co/org/repo:Q4_K_M",
        "localhost:5000/library/llama3",
        "example.com/model",
        "llama3@sha256:abc",
        "",
        ":tag",
        "a/b/c/d",
        "../x",
        "a/..",
        "a/b:..",
        "registry.ollama.ai/../..:x",
        "a/b:",
        "bad name",
        "a/b?x=1",
        "a\\b",
    ],
)
def test_parse_model_ref_returns_none_for_unsupported_names(model: str) -> None:
    assert parse_model_ref(model) is None


def test_manifest_path_uses_registry_host_layout() -> None:
    assert _REF.manifest_path == "manifests/registry.ollama.ai/library/smollm2/135m"


@pytest.mark.asyncio
async def test_download_writes_blobs_and_manifest(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server

    packets = await _collect(_client(tmp_path, server), _REF)

    assert (tmp_path / "blobs" / ("sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:"))).read_bytes() == _MODEL_BLOB
    assert (tmp_path / "blobs" / ("sha256-" + _digest(_CONFIG_BLOB).removeprefix("sha256:"))).read_bytes() == _CONFIG_BLOB
    assert (tmp_path / _REF.manifest_path).read_text() == registry.manifest_text
    assert not list((tmp_path / ".infra-downloads").glob("*"))
    assert isinstance(packets[0], PreDownloadPacket)
    assert packets[0].file_bytes_size == len(_MODEL_BLOB) + len(_CONFIG_BLOB)
    assert sum(p.downloaded_bytes_size for p in packets if isinstance(p, DownloadedPacket)) == len(_MODEL_BLOB) + len(_CONFIG_BLOB)
    assert isinstance(packets[-1], SuccessDownloadPacket)


@pytest.mark.asyncio
async def test_download_skips_blobs_already_in_store(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / ("sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:"))).write_bytes(_MODEL_BLOB)

    packets = await _collect(_client(tmp_path, server), _REF)

    assert [digest for digest, _ in registry.blob_requests] == [_digest(_CONFIG_BLOB)]
    assert sum(p.downloaded_bytes_size for p in packets if isinstance(p, DownloadedPacket)) == len(_MODEL_BLOB) + len(_CONFIG_BLOB)


@pytest.mark.asyncio
async def test_download_resumes_partial_blob(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    name = "sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:")
    (staging / (name + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    packets = await _collect(_client(tmp_path, server), _REF)

    assert (_digest(_MODEL_BLOB), "bytes=4000-") in registry.blob_requests
    assert (blobs / name).read_bytes() == _MODEL_BLOB
    total = next(p.file_bytes_size for p in packets if isinstance(p, PreDownloadPacket))
    assert sum(p.downloaded_bytes_size for p in packets if isinstance(p, DownloadedPacket)) == total


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"ignore_range": True}], indirect=True)
async def test_download_restarts_blob_when_registry_ignores_range(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    name = "sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:")
    (staging / (name + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    await _collect(_client(tmp_path, server), _REF)

    assert (blobs / name).read_bytes() == _MODEL_BLOB


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"corrupt": True}], indirect=True)
async def test_download_rejects_digest_mismatch_and_writes_no_manifest(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="verification"):
        await _collect(_client(tmp_path, server), _REF)

    assert not (tmp_path / _REF.manifest_path).exists()
    assert not list((tmp_path / ".infra-downloads").glob(_blob_name(_MODEL_BLOB) + "*"))
    assert not (tmp_path / "blobs" / ("sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:"))).exists()


@pytest.mark.asyncio
async def test_download_unknown_model_raises_404(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    _, server = registry_server
    ref = OllamaModelRef(namespace="library", name="nope", tag="latest")

    with pytest.raises(OllamaRegistryError) as exc_info:
        await _collect(_client(tmp_path, server), ref)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@patch("server.utils.ollama_registry.shutil.disk_usage")
async def test_download_fails_before_downloading_when_disk_is_full(
    mock_disk_usage: MagicMock, tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    mock_disk_usage.return_value.free = 10

    with pytest.raises(OllamaRegistryError) as exc_info:
        await _collect(_client(tmp_path, server), _REF)

    assert exc_info.value.status_code == 507
    assert registry.blob_requests == []


@pytest.mark.asyncio
async def test_download_reports_unreachable_registry(tmp_path: Path) -> None:
    client = OllamaRegistryClient(tmp_path, registry_url="http://127.0.0.1:1")

    with pytest.raises(OllamaRegistryError, match="Cannot connect"):
        await _collect(client, _REF)


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"manifest_status": 500}], indirect=True)
async def test_download_reports_manifest_server_error(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="status 500"):
        await _collect(_client(tmp_path, server), _REF)


@pytest.mark.asyncio
async def test_download_rejects_invalid_manifest(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    registry.manifest_text = '{"layers": "nope"}'

    with pytest.raises(OllamaRegistryError, match="Invalid manifest"):
        await _collect(_client(tmp_path, server), _REF)


@pytest.mark.asyncio
async def test_download_rejects_unsupported_digest_algorithm(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    registry.manifest_text = json.dumps({"config": {"digest": "md5:abc", "size": 1}, "layers": []})

    with pytest.raises(OllamaRegistryError, match="Unsupported blob digest"):
        await _collect(_client(tmp_path, server), _REF)

    assert not (tmp_path / _REF.manifest_path).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"blob_status": 503}], indirect=True)
async def test_download_reports_blob_server_error(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="status 503"):
        await _collect(_client(tmp_path, server), _REF)

    assert not (tmp_path / _REF.manifest_path).exists()


@pytest.mark.asyncio
async def test_download_discards_partial_file_larger_than_blob(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    name = "sha256-" + _digest(_MODEL_BLOB).removeprefix("sha256:")
    (staging / (name + ".partial")).write_bytes(b"x" * (len(_MODEL_BLOB) + 10))

    await _collect(_client(tmp_path, server), _REF)

    assert (_digest(_MODEL_BLOB), None) in registry.blob_requests
    assert (blobs / name).read_bytes() == _MODEL_BLOB


def _blob_name(data: bytes) -> str:
    return "sha256-" + _digest(data).removeprefix("sha256:")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "digest",
    [
        "sha256:../../etc/passwd",
        "sha256:",
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha256:" + "é" * 64,
    ],
)
async def test_download_rejects_malformed_digest_before_any_blob_request(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer], digest: str
) -> None:
    registry, server = registry_server
    registry.manifest_text = json.dumps({"config": {"digest": digest, "size": 1}, "layers": []})

    with pytest.raises(OllamaRegistryError, match="Unsupported blob digest"):
        await _collect(_client(tmp_path, server), _REF)

    assert registry.blob_requests == []
    assert not (tmp_path / _REF.manifest_path).exists()


@pytest.mark.asyncio
async def test_download_verifies_full_size_partial_without_a_request(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB)

    await _collect(_client(tmp_path, server), _REF)

    assert (_digest(_MODEL_BLOB), None) not in registry.blob_requests
    assert (blobs / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB


@pytest.mark.asyncio
async def test_download_rejects_corrupt_full_size_partial_and_next_attempt_succeeds(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(b"x" * len(_MODEL_BLOB))

    with pytest.raises(OllamaRegistryError, match="verification"):
        await _collect(_client(tmp_path, server), _REF)
    assert not (staging / (_blob_name(_MODEL_BLOB) + ".partial")).exists()

    await _collect(_client(tmp_path, server), _REF)

    assert (blobs / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"reject_range_with_416": True}], indirect=True)
async def test_download_restarts_blob_when_registry_answers_416(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    await _collect(_client(tmp_path, server), _REF)

    assert (_digest(_MODEL_BLOB), "bytes=4000-") in registry.blob_requests
    assert (_digest(_MODEL_BLOB), None) in registry.blob_requests
    assert (blobs / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"short_body": True}], indirect=True)
async def test_download_keeps_partial_when_stream_ends_early(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="ended early at 100 of"):
        await _collect(_client(tmp_path, server), _REF)

    assert (tmp_path / ".infra-downloads" / (_blob_name(_MODEL_BLOB) + ".partial")).stat().st_size == 100
    assert not (tmp_path / _REF.manifest_path).exists()


@pytest.mark.asyncio
async def test_download_does_not_count_resumed_bytes_when_registry_ignores_range(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    registry.ignore_range = True
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    packets = await _collect(_client(tmp_path, server), _REF)

    total = next(p.file_bytes_size for p in packets if isinstance(p, PreDownloadPacket))
    assert sum(p.downloaded_bytes_size for p in packets if isinstance(p, DownloadedPacket)) == total


@pytest.mark.asyncio
async def test_concurrent_downloads_sharing_a_blob_both_succeed(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    registry.gate = asyncio.Event()
    other = OllamaModelRef(namespace="library", name="smollm2", tag="360m")
    blob_locks = BlobLocks()
    first = asyncio.create_task(_collect(_client(tmp_path, server, blob_locks), _REF))
    second = asyncio.create_task(_collect(_client(tmp_path, server, blob_locks), other))

    blob_path = tmp_path / "blobs" / _blob_name(_MODEL_BLOB)
    for _ in range(400):
        lock = blob_locks.get(str(blob_path))
        if lock is not None and lock.locked() and lock._waiters:  # pyright: ignore[reportPrivateUsage]
            break
        await asyncio.sleep(0.005)
    else:
        pytest.fail("the second download never waited for the shared blob")
    registry.gate.set()
    await asyncio.gather(first, second)

    assert blob_path.read_bytes() == _MODEL_BLOB
    assert (tmp_path / _REF.manifest_path).is_file()
    assert (tmp_path / other.manifest_path).is_file()
    assert [d for d, _ in registry.blob_requests].count(_digest(_MODEL_BLOB)) == 1


@pytest.mark.asyncio
@patch("server.utils.ollama_registry._HASH_CHUNK_BYTES", 1000)
async def test_download_resumes_correctly_across_chunk_boundaries(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    packets = await _collect(_client(tmp_path, server), _REF)

    assert (blobs / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB
    assert len([p for p in packets if isinstance(p, DownloadedPacket)]) > 3


@pytest.mark.asyncio
@patch("server.utils.ollama_registry.shutil.disk_usage")
async def test_disk_space_check_counts_only_bytes_still_missing(
    mock_disk_usage: MagicMock, tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    blobs = tmp_path / "blobs"
    staging = tmp_path / ".infra-downloads"
    staging.mkdir(parents=True)
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    mock_disk_usage.return_value.free = len(_MODEL_BLOB) - 4000 + len(_CONFIG_BLOB)

    await _collect(_client(tmp_path, server), _REF)

    assert (blobs / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB


@pytest.mark.asyncio
async def test_download_writes_world_readable_files_and_leaves_no_temp_files(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server

    await _collect(_client(tmp_path, server), _REF)

    assert (tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).stat().st_mode & 0o777 == 0o644
    assert (tmp_path / _REF.manifest_path).stat().st_mode & 0o777 == 0o644
    assert not list(tmp_path.glob(".manifest-*"))


@pytest.mark.asyncio
async def test_download_keeps_partial_for_resume_after_dropped_connection_and_resumes(tmp_path: Path) -> None:
    registry = _FakeRegistry()

    async def broken_blob(request: web.Request) -> web.StreamResponse:
        if request.match_info["digest"] != _digest(_MODEL_BLOB):
            return await registry.blob(request)
        response = web.StreamResponse(headers={"Content-Length": str(len(_MODEL_BLOB))})
        await response.prepare(request)
        await response.write(_MODEL_BLOB[:100])
        await asyncio.sleep(0.1)  # let the client read the bytes before the connection goes away
        assert request.transport is not None
        request.transport.close()
        return response

    app = web.Application()
    app.router.add_get("/v2/{namespace}/{name}/manifests/{tag}", registry.manifest)
    app.router.add_get("/v2/{namespace}/{name}/blobs/{digest}", broken_blob)
    broken_server = TestServer(app)
    await broken_server.start_server()
    try:
        with pytest.raises(OllamaRegistryError, match="interrupted"):
            await _collect(_client(tmp_path, broken_server), _REF)
    finally:
        await broken_server.close()
    partial = tmp_path / ".infra-downloads" / (_blob_name(_MODEL_BLOB) + ".partial")
    assert 0 < partial.stat().st_size <= 100

    healthy_app = web.Application()
    healthy_app.router.add_get("/v2/{namespace}/{name}/manifests/{tag}", registry.manifest)
    healthy_app.router.add_get("/v2/{namespace}/{name}/blobs/{digest}", registry.blob)
    healthy = TestServer(healthy_app)
    await healthy.start_server()
    try:
        await _collect(_client(tmp_path, healthy), _REF)
    finally:
        await healthy.close()

    assert any(rng is not None and rng.startswith("bytes=") for digest, rng in registry.blob_requests if digest == _digest(_MODEL_BLOB))
    assert (tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB


async def _download_blob_then_lose_it(
    self: OllamaRegistryClient, ref: OllamaModelRef, layer: OllamaLayer
) -> AsyncGenerator[DownloadPacket]:
    async for packet in _ORIGINAL_DOWNLOAD_BLOB(self, ref, layer):  # pyright: ignore[reportPrivateUsage]
        yield packet
    (self._staging_dir / ("sha256-" + layer.digest.removeprefix("sha256:"))).unlink()  # pyright: ignore[reportPrivateUsage]


_ORIGINAL_DOWNLOAD_BLOB = OllamaRegistryClient._download_blob  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
@patch.object(OllamaRegistryClient, "_download_blob", _download_blob_then_lose_it)
async def test_download_writes_no_manifest_when_a_blob_disappears(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="disappeared"):
        await _collect(_client(tmp_path, server), _REF)

    assert not (tmp_path / _REF.manifest_path).exists()


@pytest.mark.asyncio
async def test_download_replaces_a_manifest_from_an_earlier_install(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    manifest_path = tmp_path / _REF.manifest_path
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("stale")

    await _collect(_client(tmp_path, server), _REF)

    assert manifest_path.read_text() == registry.manifest_text


@pytest.mark.asyncio
async def test_download_rejects_manifest_with_negative_layer_size(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    registry.manifest_text = json.dumps({"config": {"digest": _digest(_CONFIG_BLOB), "size": -5}, "layers": []})

    with pytest.raises(OllamaRegistryError, match="Invalid manifest"):
        await _collect(_client(tmp_path, server), _REF)

    assert registry.blob_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"short_body": True}], indirect=True)
async def test_download_keeps_partial_out_of_the_blobs_dir(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    _, server = registry_server

    with pytest.raises(OllamaRegistryError, match="ended early"):
        await _collect(_client(tmp_path, server), _REF)

    assert not list((tmp_path / "blobs").glob("*.partial"))
    assert (tmp_path / ".infra-downloads" / (_blob_name(_MODEL_BLOB) + ".partial")).is_file()


@pytest.mark.asyncio
async def test_download_keeps_finished_blob_out_of_the_blobs_dir_until_the_manifest_is_written(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    seen_in_blobs: list[bool] = []

    async for packet in _client(tmp_path, server).download(_REF):
        if isinstance(packet, SuccessDownloadPacket):
            break
        seen_in_blobs.append((tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).exists())

    assert seen_in_blobs
    assert not any(seen_in_blobs)


@pytest.mark.asyncio
async def test_download_uses_staged_blob_without_a_request(tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]) -> None:
    registry, server = registry_server
    staging = tmp_path / ".infra-downloads"
    staging.mkdir()
    (staging / _blob_name(_MODEL_BLOB)).write_bytes(_MODEL_BLOB)

    await _collect(_client(tmp_path, server), _REF)

    assert [digest for digest, _ in registry.blob_requests] == [_digest(_CONFIG_BLOB)]
    assert (tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB
    assert not (staging / _blob_name(_MODEL_BLOB)).exists()


@pytest.mark.asyncio
async def test_download_refreshes_the_time_of_a_blob_moved_into_the_store(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    staging = tmp_path / ".infra-downloads"
    staging.mkdir()
    staged = staging / _blob_name(_MODEL_BLOB)
    staged.write_bytes(_MODEL_BLOB)
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(staged, (two_days_ago, two_days_ago))

    await _collect(_client(tmp_path, server), _REF)

    assert (tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).stat().st_mtime > time.time() - 60


@pytest.mark.asyncio
async def test_download_removes_abandoned_staged_files_and_keeps_recent_ones(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    staging = tmp_path / ".infra-downloads"
    staging.mkdir()
    old_partial = staging / ("sha256-" + "1" * 64 + ".partial")
    old_blob = staging / ("sha256-" + "2" * 64)
    recent_partial = staging / ("sha256-" + "3" * 64 + ".partial")
    for path in (old_partial, old_blob, recent_partial):
        path.write_bytes(b"x")
    long_ago = time.time() - 30 * 24 * 60 * 60
    os.utime(old_partial, (long_ago, long_ago))
    os.utime(old_blob, (long_ago, long_ago))

    await _collect(_client(tmp_path, server), _REF)

    assert not old_partial.exists()
    assert not old_blob.exists()
    assert recent_partial.exists()


@pytest.mark.asyncio
async def test_download_does_not_remove_an_old_staged_file_that_is_being_downloaded(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    _, server = registry_server
    staging = tmp_path / ".infra-downloads"
    staging.mkdir()
    partial = staging / (_blob_name(_MODEL_BLOB) + ".partial")
    partial.write_bytes(b"x")
    long_ago = time.time() - 30 * 24 * 60 * 60
    os.utime(partial, (long_ago, long_ago))
    blob_locks = BlobLocks()
    lock = asyncio.Lock()
    blob_locks[str(tmp_path / "blobs" / _blob_name(_MODEL_BLOB))] = lock
    client = _client(tmp_path, server, blob_locks)
    (tmp_path / "blobs").mkdir()

    async with lock:
        client._sweep_staging()  # pyright: ignore[reportPrivateUsage]

    assert partial.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("registry_server", [{"redirect_to_cdn": True}], indirect=True)
async def test_download_follows_the_registry_redirect_and_keeps_the_range_header_when_resuming(
    tmp_path: Path, registry_server: tuple[_FakeRegistry, TestServer]
) -> None:
    registry, server = registry_server
    staging = tmp_path / ".infra-downloads"
    staging.mkdir()
    (staging / (_blob_name(_MODEL_BLOB) + ".partial")).write_bytes(_MODEL_BLOB[:4000])

    await _collect(_client(tmp_path, server), _REF)

    assert (_digest(_MODEL_BLOB), "bytes=4000-") in registry.blob_requests
    assert (tmp_path / "blobs" / _blob_name(_MODEL_BLOB)).read_bytes() == _MODEL_BLOB
