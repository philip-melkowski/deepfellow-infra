# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Client that downloads Ollama registry models straight into an Ollama model store.

Ollama keeps models as `manifests/<host>/<namespace>/<name>/<tag>` plus content-addressed `blobs/sha256-<hex>`
files. Writing those files from outside the container lets a model be installed while the Ollama container
itself has no access to the registry.

Ollama deletes `blobs/` files no manifest references (and `.partial` files) after about an hour, whenever an instance
that does not disable pruning starts. So downloads are staged in `.infra-downloads/`, which Ollama does not scan, and
only moved into `blobs/` right before the manifest that references them is written.
"""

import asyncio
import hashlib
import os
import re
import shutil
import time
import weakref
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from uuid import uuid4

import aiofiles
import aiohttp
from aiohttp import ClientTimeout
from pydantic import BaseModel, Field, ValidationError

from server.utils.core import DownloadedPacket, DownloadPacket, PreDownloadPacket, SuccessDownloadPacket
from server.utils.logger import logger

OLLAMA_REGISTRY_HOST = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"
_DEFAULT_TAG = "latest"
_MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
_MANIFEST_TIMEOUT_SECONDS = 60
_BLOB_CONNECT_TIMEOUT_SECONDS = 60
_BLOB_READ_TIMEOUT_SECONDS = 300  # a blob download that receives nothing for this long is treated as interrupted
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_SHA256_PREFIX = "sha256:"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
# Ollama's own name rules; they also keep a name from escaping the manifests directory or the registry URL.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TAG_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$")
_PARTIAL_SUFFIX = ".partial"
_STAGING_DIRNAME = ".infra-downloads"
_STAGING_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # staged files untouched for this long belong to abandoned downloads
_FILE_MODE = 0o644


class OllamaRegistryError(Exception):
    """Raised when a model cannot be downloaded from the Ollama registry."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OllamaModelRef(BaseModel):
    """Model name split into the parts that make up its manifest path."""

    namespace: str
    name: str
    tag: str

    @property
    def manifest_path(self) -> str:
        """Return the manifest location relative to the `models` directory."""
        return f"manifests/{OLLAMA_REGISTRY_HOST}/{self.namespace}/{self.name}/{self.tag}"


class OllamaLayer(BaseModel):
    digest: str
    size: int = Field(ge=0)


class OllamaManifest(BaseModel):
    config: OllamaLayer
    layers: list[OllamaLayer]

    def all_layers(self) -> list[OllamaLayer]:
        """Return the config blob followed by the layer blobs."""
        return [self.config, *self.layers]


def parse_model_ref(model: str) -> OllamaModelRef | None:
    """Parse `[namespace/]name[:tag]` into a reference on the default registry.

    Return None for names this client does not handle: a name pointing at another registry (for example
    `hf.co/org/repo`, `localhost/x` or `host:5000/x`), a digest reference, or a name that is not a valid Ollama
    name. The host `registry.ollama.ai` is accepted only as a prefix of the full `namespace/name` form.
    """
    if not model or "@" in model:
        return None
    parts = model.split("/")
    if len(parts) > 3 or (len(parts) == 3 and parts[0] != OLLAMA_REGISTRY_HOST):
        return None
    if len(parts) == 2 and (parts[0] == "localhost" or "." in parts[0] or ":" in parts[0]):
        return None
    if len(parts) == 3:
        parts = parts[1:]
    namespace, name_with_tag = (parts[0], parts[1]) if len(parts) == 2 else (_DEFAULT_NAMESPACE, parts[0])
    name, has_tag, tag = name_with_tag.partition(":")
    if not _NAME_RE.match(namespace) or not _NAME_RE.match(name) or (has_tag and not _TAG_RE.match(tag)):
        return None
    return OllamaModelRef(namespace=namespace, name=name, tag=tag or _DEFAULT_TAG)


def _blob_filename(digest: str) -> str:
    """Return the store file name of a blob; only a well-formed sha256 digest is accepted, so it can't escape `blobs/`."""
    if not _DIGEST_RE.match(digest):
        msg = f"Unsupported blob digest in the model manifest: {digest!r}"
        raise OllamaRegistryError(msg)
    return "sha256-" + digest[len(_SHA256_PREFIX) :]


def _describe(error: Exception) -> str:
    """Return the error with its type, since `str()` of a timeout is empty."""
    return f"{type(error).__name__}: {error}"


class _Sha256:
    """Running sha256 that can be restarted when a blob is downloaded from scratch."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def update(self, data: bytes) -> None:
        """Feed more bytes."""
        self._hash.update(data)

    def reset(self) -> None:
        """Start over."""
        self._hash = hashlib.sha256()

    def hexdigest(self) -> str:
        """Return the digest of the bytes fed so far."""
        return self._hash.hexdigest()


def _hash_file(path: Path, hasher: _Sha256) -> None:
    """Feed the whole file to `hasher`, so a resumed download continues the digest of its partial file."""
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)


BlobLocks = weakref.WeakValueDictionary[str, asyncio.Lock]
"""Per-blob locks keyed by blob path, shared by all clients that write to the same model store."""


class OllamaRegistryClient:
    """Download models from the Ollama registry into a model store (the `models` directory of `~/.ollama`)."""

    def __init__(
        self,
        models_dir: Path,
        registry_url: str = f"https://{OLLAMA_REGISTRY_HOST}",
        blob_locks: BlobLocks | None = None,
    ) -> None:
        # Two models can share a blob (license, params, ...) and would otherwise write the same `.partial` file,
        # so downloads that may run concurrently must pass the same `blob_locks`.
        self._blob_locks = blob_locks if blob_locks is not None else BlobLocks()
        self._models_dir = models_dir
        self._blobs_dir = models_dir / "blobs"
        self._staging_dir = models_dir / _STAGING_DIRNAME
        self._registry_url = registry_url.rstrip("/")

    def _blob_lock(self, layer: OllamaLayer) -> asyncio.Lock:
        key = str(self._blobs_dir / _blob_filename(layer.digest))
        lock = self._blob_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._blob_locks[key] = lock
        return lock

    async def download(self, ref: OllamaModelRef) -> AsyncGenerator[DownloadPacket]:
        """Download manifest and blobs of the model, writing the manifest last.

        Yield a `PreDownloadPacket` with the total size, `DownloadedPacket`s as bytes arrive (blobs already in the
        store or staged count as downloaded) and a final `SuccessDownloadPacket` pointing at the manifest.
        """
        manifest_text, manifest = await self._fetch_manifest(ref)
        layers = manifest.all_layers()
        self._blobs_dir.mkdir(parents=True, exist_ok=True)
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_staging()

        missing: list[OllamaLayer] = []
        present_bytes = 0
        for layer in layers:
            if self._is_blob_present(layer):
                present_bytes += layer.size
            else:
                missing.append(layer)
        self._ensure_free_space(missing)

        yield PreDownloadPacket(sum(layer.size for layer in layers))
        if present_bytes:
            yield DownloadedPacket(present_bytes)
        for layer in missing:
            async with aclosing(self._download_blob(ref, layer)) as packets:
                async for packet in packets:
                    yield packet

        for layer in layers:
            await self._publish_blob(layer)
        manifest_path = self._write_manifest(ref, manifest_text)
        yield SuccessDownloadPacket(manifest_path, ref.manifest_path)

    async def _fetch_manifest(self, ref: OllamaModelRef) -> tuple[str, OllamaManifest]:
        url = f"{self._registry_url}/v2/{ref.namespace}/{ref.name}/manifests/{ref.tag}"
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, headers={"Accept": _MANIFEST_ACCEPT}, timeout=ClientTimeout(_MANIFEST_TIMEOUT_SECONDS)) as response,
            ):
                body = await response.text(errors="replace")
                if response.status == 404:
                    msg = f"Model {ref.namespace}/{ref.name}:{ref.tag} not found in the Ollama registry"
                    raise OllamaRegistryError(msg, 404)
                if response.status != 200:
                    msg = f"Ollama registry returned status {response.status} for {url}: {body[:200]}"
                    raise OllamaRegistryError(msg)
        except (aiohttp.ClientError, TimeoutError) as e:
            msg = f"Cannot connect to the Ollama registry: {_describe(e)}"
            raise OllamaRegistryError(msg) from e
        try:
            return body, OllamaManifest.model_validate_json(body)
        except ValidationError as e:
            msg = f"Invalid manifest received from the Ollama registry for {url}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}"
            raise OllamaRegistryError(msg) from e

    def _is_blob_present(self, layer: OllamaLayer) -> bool:
        """Tell whether the complete blob is in the store or staged."""
        filename = _blob_filename(layer.digest)
        return any(
            path.is_file() and path.stat().st_size == layer.size for path in (self._blobs_dir / filename, self._staging_dir / filename)
        )

    def _sweep_staging(self) -> None:
        """Delete staged files of abandoned downloads: untouched for a week and not being downloaded right now."""
        cutoff = time.time() - _STAGING_MAX_AGE_SECONDS
        for path in self._staging_dir.iterdir():
            name = path.name.removesuffix(_PARTIAL_SUFFIX)
            lock = self._blob_locks.get(str(self._blobs_dir / name))
            if path.is_file() and (lock is None or not lock.locked()) and path.stat().st_mtime < cutoff:
                logger.info("Removing abandoned staged download %s", path)
                path.unlink(missing_ok=True)

    async def _publish_blob(self, layer: OllamaLayer) -> None:
        """Move a staged blob into `blobs/`, where the manifest written next makes Ollama see it."""
        filename = _blob_filename(layer.digest)
        final_path = self._blobs_dir / filename
        staged_path = self._staging_dir / filename
        async with self._blob_lock(layer):
            if staged_path.is_file() and staged_path.stat().st_size == layer.size:
                staged_path.chmod(_FILE_MODE)
                staged_path.replace(final_path)
                os.utime(final_path)  # a move keeps the old time, and a fresh file is not pruned right away
            elif not (final_path.is_file() and final_path.stat().st_size == layer.size):
                # Something removed a finished blob while the model was downloading.
                msg = f"Blob {layer.digest} disappeared from the Ollama store before the model was complete"
                raise OllamaRegistryError(msg)

    def _ensure_free_space(self, missing: list[OllamaLayer]) -> None:
        needed = 0
        for layer in missing:
            partial = self._staging_dir / (_blob_filename(layer.digest) + _PARTIAL_SUFFIX)
            already = partial.stat().st_size if partial.is_file() else 0
            needed += max(layer.size - already, 0)
        free = shutil.disk_usage(self._blobs_dir).free
        if needed > free:
            msg = f"Not enough disk space to download the model: {needed} bytes needed, {free} bytes free"
            raise OllamaRegistryError(msg, 507)

    async def _download_blob(self, ref: OllamaModelRef, layer: OllamaLayer) -> AsyncGenerator[DownloadPacket]:
        staged_path = self._staging_dir / _blob_filename(layer.digest)
        partial_path = staged_path.with_name(staged_path.name + _PARTIAL_SUFFIX)
        url = f"{self._registry_url}/v2/{ref.namespace}/{ref.name}/blobs/{layer.digest}"

        async with self._blob_lock(layer):
            if self._is_blob_present(layer):
                # Another download of a model sharing this blob finished while we waited.
                yield DownloadedPacket(layer.size)
                return

            offset = partial_path.stat().st_size if partial_path.is_file() else 0
            if offset > layer.size:
                partial_path.unlink()
                offset = 0
            hasher = _Sha256()
            if offset:
                logger.info("Resuming blob %s at %d of %d bytes", layer.digest, offset, layer.size)
                await asyncio.to_thread(_hash_file, partial_path, hasher)
            if offset == layer.size:
                # Everything was written before the last attempt stopped; only verification is left.
                yield DownloadedPacket(offset)
            else:
                async with aclosing(self._fetch_blob(url, partial_path, offset, hasher, layer)) as packets:
                    async for packet in packets:
                        yield packet

            written = partial_path.stat().st_size
            if written < layer.size:
                # Keep the partial file: the next attempt resumes it.
                msg = f"Download of blob {layer.digest} ended early at {written} of {layer.size} bytes"
                raise OllamaRegistryError(msg)
            if written > layer.size or f"{_SHA256_PREFIX}{hasher.hexdigest()}" != layer.digest:
                partial_path.unlink(missing_ok=True)
                msg = f"Blob {layer.digest} failed verification after download (size or sha256 mismatch)"
                raise OllamaRegistryError(msg)

            partial_path.replace(staged_path)

    async def _fetch_blob(
        self, url: str, partial_path: Path, offset: int, hasher: _Sha256, layer: OllamaLayer
    ) -> AsyncGenerator[DownloadPacket]:
        """Stream a blob into its partial file, continuing from `offset` when the registry honors `Range`."""
        timeout = ClientTimeout(total=None, sock_connect=_BLOB_CONNECT_TIMEOUT_SECONDS, sock_read=_BLOB_READ_TIMEOUT_SECONDS)
        try:
            while True:  # runs twice at most: a 416 resets the offset to 0, which can no longer be answered with 416
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, timeout=timeout) as response,
                ):
                    if response.status == 416 and offset:
                        # The partial file doesn't fit the blob: start over.
                        partial_path.unlink(missing_ok=True)
                        hasher.reset()
                        offset = 0
                        continue
                    if response.status == 200 and offset:
                        # The registry ignored the Range header: start the blob over.
                        hasher.reset()
                        offset = 0
                    elif response.status == 206:
                        yield DownloadedPacket(offset)
                    elif response.status != 200:
                        body = await response.text(errors="replace")
                        msg = f"Ollama registry returned status {response.status} for blob {layer.digest}: {body[:200]}"
                        raise OllamaRegistryError(msg)
                    async with aiofiles.open(partial_path, "ab" if offset else "wb") as f:
                        async for chunk in response.content.iter_chunked(_HASH_CHUNK_BYTES):
                            hasher.update(chunk)
                            await f.write(chunk)
                            yield DownloadedPacket(len(chunk))
                    return
        except (aiohttp.ClientError, TimeoutError) as e:
            # The partial file is kept so the next attempt can resume it.
            msg = f"Download of blob {layer.digest} was interrupted: {_describe(e)}"
            raise OllamaRegistryError(msg) from e

    def _write_manifest(self, ref: OllamaModelRef, manifest_text: str) -> Path:
        manifest_path = self._models_dir / ref.manifest_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._models_dir / f".manifest-{uuid4().hex}.tmp"
        try:
            temp_path.write_text(manifest_text, encoding="utf-8")
            temp_path.chmod(_FILE_MODE)
            temp_path.replace(manifest_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return manifest_path
