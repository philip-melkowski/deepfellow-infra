# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Snapshot of the real tag lists of the images behind the llama.cpp, vLLM, SGLang and Ollama services.

`data/real_docker_tags.json` holds a trimmed copy of the lists Infra's registry client returned (GHCR and Docker
Hub, 2026-10-13), before the services' `filter_docker_tags`: about 200 tags, enough to cover every variant the
tests use (see `_note` in the file). Tests use them so tag selection is checked against real tag formats, not
invented ones. To refresh, save `get_tags()` output for each image and trim it the same way; `just test-integration`
checks the full, live lists.
"""

import json
from pathlib import Path
from typing import Literal

_DATA = Path(__file__).parent / "data" / "real_docker_tags.json"


def real_tags(service: Literal["llamacpp", "vllm", "sglang", "ollama"]) -> list[str]:
    """Return the snapshot tag list for the service's image."""
    return json.loads(_DATA.read_text())[service]
