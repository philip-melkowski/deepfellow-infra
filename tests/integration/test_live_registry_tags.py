# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Integration tests: tag selection against the live registries (GHCR, Docker Hub) and the host's NVIDIA driver.

They need network access (and Docker with a GPU for the CUDA check), so they are skipped unless
`DF_TEST_INTEGRATION_REGISTRIES=1` is set. Run them with `just test-integration`. They catch what the snapshot-based unit
tests cannot: a registry changing its tag format or listing, or a pinned default image being removed.
"""

import os
import re

import pytest

from server.services.llamacpp_service import _const as llamacpp_const  # pyright: ignore[reportPrivateUsage]
from server.services.ollama_service import _const as ollama_const  # pyright: ignore[reportPrivateUsage]
from server.services.sglang_service import _const as sglang_const  # pyright: ignore[reportPrivateUsage]
from server.services.vllm_service import _const as vllm_const  # pyright: ignore[reportPrivateUsage]
from server.utils.docker_image_version import newest_docker_tag, split_image_repo_tag
from server.utils.hardware import get_nvidia_max_cuda_version
from server.utils.registry_client import image_without_registry_prefix, registry_for

pytestmark = [
    pytest.mark.skipif(os.environ.get("DF_TEST_INTEGRATION_REGISTRIES") != "1", reason="set DF_TEST_INTEGRATION_REGISTRIES=1 to run"),
]

PINNED_IMAGES = {
    "llamacpp-gpu": llamacpp_const.images["gpu"].name,
    "llamacpp-vulkan": llamacpp_const.images["vulkan"].name,
    "llamacpp-cpu": llamacpp_const.images["cpu"].name,
    "vllm-gpu": vllm_const.images["gpu"].name,
    "sglang": sglang_const.image.name,
    "ollama": ollama_const.image.name,
}


@pytest.mark.parametrize("image", PINNED_IMAGES.values(), ids=PINNED_IMAGES.keys())
@pytest.mark.asyncio
async def test_pinned_image_still_exists_and_has_a_selectable_newest_tag(image: str) -> None:
    repo, pinned = split_image_repo_tag(image)
    client = registry_for(repo)
    name = image_without_registry_prefix(repo)

    assert await client.tag_exists(name, pinned), f"pinned image {image} no longer exists in its registry"

    tags = await client.get_tags(name)
    assert tags, f"no tags returned for {repo}"
    newest = newest_docker_tag([*tags, pinned], pinned)
    assert newest is not None, f"pinned tag {pinned!r} has an unrecognized format"


@pytest.mark.asyncio
async def test_llamacpp_tag_list_reaches_builds_at_least_as_new_as_the_pinned_one() -> None:
    """GHCR lists tags oldest first, so a list cut short stops at old builds (it once ended at b5985, below the pinned b10902)."""
    repo, pinned = split_image_repo_tag(llamacpp_const.images["gpu"].name)
    tags = await registry_for(repo).get_tags(image_without_registry_prefix(repo))

    pinned_build = int(re.search(r"-b(\d+)$", pinned)[1])  # pyright: ignore[reportOptionalSubscript]
    builds = [int(m[1]) for tag in tags if (m := re.search(r"-b(\d+)$", tag))]
    assert max(builds) >= pinned_build


@pytest.mark.asyncio
async def test_host_cuda_version_is_none_or_a_version_pair() -> None:
    version = await get_nvidia_max_cuda_version()
    assert version is None or (len(version) == 2 and version[0] >= 10)
