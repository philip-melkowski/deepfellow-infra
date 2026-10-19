# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Router options."""

from typing import Annotated, Any

from pydantic import BaseModel, Field


class RouterRules(BaseModel):
    """Layer 1: cheap rules checked before any embedding is computed."""

    force_large_over_chars: Annotated[
        int | None, Field(ge=1, description="Send to the large model when the prompt is longer than this")
    ] = None
    force_large_keywords: Annotated[
        list[str], Field(description="Send to the large model when the prompt contains any of these (case-insensitive)")
    ] = []


class RouterOptions(BaseModel):
    """Options of the router service (stored as service options JSON)."""

    small_model: Annotated[str, Field(min_length=1, description="Model used for SIMPLE requests")]
    large_model: Annotated[str, Field(min_length=1, description="Model used for COMPLEX requests and as fallback")]
    embedding_model: Annotated[str, Field(min_length=1, description="Embedding model used by layer 2")] = "nomic-embed-text"
    embedding_prefix: Annotated[str, Field(description="Prefix prepended to the prompt before embedding")] = "classification: "
    index_path: Annotated[str, Field(min_length=1, description="Path to the .npz index with vectors and labels")]
    k: Annotated[int, Field(ge=1, description="Number of neighbours")] = 20
    threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Score at or above which the large model is used")] = 0.3
    rules: RouterRules = RouterRules()

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "RouterOptions":
        """Build options from the flat service form (keywords as a comma separated string, numbers possibly as strings)."""
        data = {k: v for k, v in spec.items() if v not in (None, "") and k not in {"force_large_over_chars", "force_large_keywords"}}
        keywords = spec.get("force_large_keywords") or []
        if isinstance(keywords, str):
            keywords = [w.strip() for w in keywords.split(",") if w.strip()]
        chars = spec.get("force_large_over_chars")
        data["rules"] = {"force_large_over_chars": chars if chars not in (None, "") else None, "force_large_keywords": keywords}
        return cls.model_validate(data)
