# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Router decision engine: rules (layer 1), then kNN on embeddings (layer 2), with a safe fallback."""

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel

from server.router.config import RouterOptions
from server.router.index import KnnIndex

logger = logging.getLogger("uvicorn.error")

type Embedder = Callable[[str], Awaitable[Sequence[float]]]


class RouterDecision(BaseModel):
    model: str
    layer: Literal["rules", "knn", "fallback"]
    reason: str
    score: float | None = None
    latency_ms: float


class RouterEngine:
    def __init__(self, options: RouterOptions, index: KnnIndex, embed: Embedder) -> None:
        self._options = options
        self._index = index
        self._embed = embed
        self._keywords = [w.lower() for w in options.rules.force_large_keywords]

    def _rule(self, prompt: str) -> str | None:
        rules = self._options.rules
        if rules.force_large_over_chars is not None and len(prompt) > rules.force_large_over_chars:
            return f"prompt longer than {rules.force_large_over_chars} characters"
        lowered = prompt.lower()
        for word in self._keywords:
            if word in lowered:
                return f"keyword {word!r}"
        return None

    async def decide(self, prompt: str) -> RouterDecision:
        """Choose the model for the prompt; any failure of the classifier falls back to the large model."""
        start = time.perf_counter()

        def done(model: str, layer: Literal["rules", "knn", "fallback"], reason: str, score: float | None = None) -> RouterDecision:
            decision = RouterDecision(model=model, layer=layer, reason=reason, score=score, latency_ms=(time.perf_counter() - start) * 1000)
            logger.info("router decision: %s", decision.model_dump_json())
            return decision

        try:
            rule_reason = self._rule(prompt)
            if rule_reason is not None:
                return done(self._options.large_model, "rules", rule_reason)
            vector = await self._embed(self._options.embedding_prefix + prompt)
            score = self._index.score(vector, self._options.k)
        except Exception as e:
            logger.warning("router classifier failed, using the large model: %s", e)
            return done(self._options.large_model, "fallback", f"{type(e).__name__}: {e}")
        if score >= self._options.threshold:
            return done(self._options.large_model, "knn", f"score {score:.3f} >= threshold {self._options.threshold}", score)
        return done(self._options.small_model, "knn", f"score {score:.3f} < threshold {self._options.threshold}", score)
