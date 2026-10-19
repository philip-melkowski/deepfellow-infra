# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for filter_docker_tags in llama.cpp and vLLM services."""

from unittest.mock import MagicMock, patch

import pytest

from server.services.llamacpp_service import LLamacppService
from server.services.vllm_service import VllmService

LLAMACPP_TAGS = [
    "server-cuda-b7836",
    "server-cuda-b7000",
    "server-vulkan-b7836",
    "server-vulkan-b7000",
    "server-b7836",
    "server-b7000",
    "full-cuda-b7836",
    "light-b7836",
]

VLLM_TAGS = [
    "v0.19.0-cu130-ubuntu2404",
    "v0.19.0-cpu",
    "v0.18.0-cu128-ubuntu2204",
    "v0.18.0-cpu",
    "v0.17.0-cu120",
]

VLLM_CPU_TAGS = ["latest", "latest-arm64", "latest-x86_64", "v0.31.0", "v0.31.0-arm64", "v0.31.0-x86_64"]


@pytest.fixture
def llamacpp_service() -> LLamacppService:
    svc: LLamacppService = MagicMock(spec=LLamacppService)
    svc.filter_docker_tags = LLamacppService.filter_docker_tags.__get__(svc, LLamacppService)  # type: ignore[method-assign]
    svc._image_variant_for_hardware = LLamacppService._image_variant_for_hardware.__get__(svc, LLamacppService)  # type: ignore[method-assign]
    return svc


@pytest.fixture
def vllm_service() -> VllmService:
    svc: VllmService = MagicMock(spec=VllmService)
    svc.filter_docker_tags = VllmService.filter_docker_tags.__get__(svc, VllmService)  # type: ignore[method-assign]
    svc._use_gpu_image = VllmService._use_gpu_image.__get__(svc, VllmService)  # type: ignore[method-assign]
    svc._is_arm_host = VllmService._is_arm_host  # type: ignore[method-assign]
    return svc


class TestLlamaCppFilterDockerTags:
    def test_gpu_returns_cuda_tags(self, llamacpp_service: LLamacppService) -> None:
        result = llamacpp_service.filter_docker_tags(LLAMACPP_TAGS, "GPU")
        assert all("cuda" in t.lower() for t in result)
        assert "server-cuda-b7836" in result
        assert "server-b7836" not in result
        assert "server-vulkan-b7836" not in result
        assert "full-cuda-b7836" not in result

    def test_cpu_excludes_cuda_and_vulkan(self, llamacpp_service: LLamacppService) -> None:
        result = llamacpp_service.filter_docker_tags(LLAMACPP_TAGS, "cpu")
        assert all("cuda" not in t.lower() and "vulkan" not in t.lower() for t in result)
        assert "server-b7836" in result
        assert "server-cuda-b7836" not in result
        assert "light-b7836" not in result

    def test_vulkan_returns_vulkan_tags(self, llamacpp_service: LLamacppService) -> None:
        result = llamacpp_service.filter_docker_tags(LLAMACPP_TAGS, "vulkan")
        assert all("vulkan" in t.lower() for t in result)
        assert "server-vulkan-b7836" in result

    def test_none_hardware_returns_all(self, llamacpp_service: LLamacppService) -> None:
        result = llamacpp_service.filter_docker_tags(LLAMACPP_TAGS, None)
        assert result == LLAMACPP_TAGS

    def test_intel_hardware_returns_vulkan_tags(self, llamacpp_service: LLamacppService) -> None:
        result = llamacpp_service.filter_docker_tags(LLAMACPP_TAGS, "intel")
        assert all("vulkan" in t.lower() for t in result)
        assert "server-vulkan-b7836" in result
        assert "server-cuda-b7836" not in result


class TestVllmFilterDockerTags:
    def test_gpu_returns_cuda_tags(self, vllm_service: VllmService) -> None:
        result = vllm_service.filter_docker_tags(VLLM_TAGS, "GPU")
        assert all("cu" in t.lower() and "cpu" not in t.lower() for t in result)
        assert "v0.19.0-cu130-ubuntu2404" in result
        assert "v0.19.0-cpu" not in result

    def test_gpu_keeps_plain_release_tags(self, vllm_service: VllmService) -> None:
        tags = ["v0.31.0", "v0.31.0-ubuntu2404", "v0.31.0-x86_64", "v0.31.0-x86_64-zstd", "v0.31.0-cu129", "v0.31.0-cpu", "latest"]
        result = vllm_service.filter_docker_tags(tags, "GPU")
        assert result == ["v0.31.0", "v0.31.0-ubuntu2404", "v0.31.0-x86_64", "v0.31.0-cu129"]

    def test_cpu_on_x86_excludes_arm64_tags(self, vllm_service: VllmService) -> None:
        with patch("platform.machine", return_value="x86_64"):
            result = vllm_service.filter_docker_tags(VLLM_CPU_TAGS, "cpu")
        assert result == ["latest", "latest-x86_64", "v0.31.0", "v0.31.0-x86_64"]

    @pytest.mark.parametrize("machine", ["arm64", "aarch64"])
    def test_cpu_on_arm_excludes_x86_tags(self, vllm_service: VllmService, machine: str) -> None:
        with patch("platform.machine", return_value=machine):
            result = vllm_service.filter_docker_tags(VLLM_CPU_TAGS, "cpu")
        assert result == ["latest", "latest-arm64", "v0.31.0", "v0.31.0-arm64"]

    def test_none_hardware_returns_all(self, vllm_service: VllmService) -> None:
        result = vllm_service.filter_docker_tags(VLLM_TAGS, None)
        assert result == VLLM_TAGS
