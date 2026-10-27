# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Router decision engine: rules and heuristics (layer 1), kNN on embeddings (layer 2), a decision model (layer 3), safe fallback."""

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel

from server.router.config import RouterOptions
from server.router.decider import Decider
from server.router.heuristics import HeuristicScorer
from server.router.index import KnnIndex

logger = logging.getLogger("uvicorn.error")

type Embedder = Callable[[str], Awaitable[Sequence[float]]]
type Layer = Literal["rules", "heuristics", "knn", "decider", "fallback"]


class RouterDecision(BaseModel):
    model: str
    layer: Layer
    reason: str
    score: float | None = None
    heuristic: float | None = None
    decider_score: float | None = None
    latency_ms: float


class RouterEngine:
    def __init__(
        self,
        options: RouterOptions,
        index: KnnIndex,
        embed: Embedder,
        heuristics: HeuristicScorer | None = None,
        decider: Decider | None = None,
    ) -> None:
        self._options = options
        self._index = index
        self._embed = embed
        self._heuristics = heuristics
        self._decider = decider
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

        def done(
            model: str,
            layer: Layer,
            reason: str,
            score: float | None = None,
            heuristic: float | None = None,
            decider_score: float | None = None,
        ) -> RouterDecision:
            decision = RouterDecision(
                model=model,
                layer=layer,
                reason=reason,
                score=score,
                heuristic=heuristic,
                decider_score=decider_score,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
            logger.info("router decision: %s", decision.model_dump_json())
            return decision

        try:
            rule_reason = self._rule(prompt)
            if rule_reason is not None:
                return done(self._options.large_model, "rules", rule_reason)
            heuristic = self._heuristics.points(prompt) if self._heuristics is not None else None
            if heuristic is not None and heuristic >= self._options.heuristic_threshold:
                reason = f"heuristic points {heuristic:g} >= {self._options.heuristic_threshold:g}"
                return done(self._options.large_model, "heuristics", reason, heuristic=heuristic)
            vector = await self._embed(self._options.embedding_prefix + prompt)
            score = self._index.score(vector, self._options.k)
        except Exception as e:
            logger.warning("router classifier failed, using the large model: %s", e)
            return done(self._options.large_model, "fallback", f"{type(e).__name__}: {e}")
        threshold = self._options.threshold
        if self._decider is not None and self._decider.in_zone(score):
            try:
                p = await self._decider.score(prompt, score)
            except Exception as e:
                logger.warning("router decision model failed, using the large model: %s", e)
                return done(self._options.large_model, "fallback", f"decision model: {type(e).__name__}: {e}", score, heuristic)
            model = self._options.large_model if p >= threshold else self._options.small_model
            sign = ">=" if p >= threshold else "<"
            return done(model, "decider", f"decision score {p:.3f} {sign} threshold {threshold}", score, heuristic, p)
        if score >= threshold:
            return done(self._options.large_model, "knn", f"score {score:.3f} >= threshold {threshold}", score, heuristic)
        return done(self._options.small_model, "knn", f"score {score:.3f} < threshold {threshold}", score, heuristic)
