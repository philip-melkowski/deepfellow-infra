# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Layer 3: a decision model answers yes/no questions about the prompt; a logistic regression joins them with the kNN score."""

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator

from server.models.api import SystemOneNoulQuestion

type AskDecisionModel = Callable[[str, str, dict[str, SystemOneNoulQuestion]], Awaitable[dict[str, float]]]
"""Sends (model, prompt, questions) to the decision model and returns P(yes) for every question."""

_EPS = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


class DeciderParams(BaseModel):
    """Parameters of layer 3, fitted offline on labelled prompts (stored as a JSON file)."""

    model: Annotated[str, Field(min_length=1, description="Decision model served on /v1/systemone")]
    questions: Annotated[dict[str, SystemOneNoulQuestion], Field(min_length=1, description="Yes/no questions asked about the prompt")]
    intercept: float
    knn_weight: Annotated[float, Field(description="Weight of logit(kNN score)")]
    weights: Annotated[dict[str, float], Field(description="Weight of logit(P(yes)) for every question")]
    zone_low: Annotated[float, Field(ge=0.0, le=1.0, description="Layer 3 is asked only for kNN scores in [zone_low, zone_high)")] = 0.2
    zone_high: Annotated[float, Field(ge=0.0, le=1.0)] = 0.45
    timeout_seconds: Annotated[float, Field(gt=0.0, description="Time limit of the decision model call")] = 5.0

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.weights) != set(self.questions):
            raise ValueError("weights must have exactly one entry per question")
        if self.zone_low >= self.zone_high:
            raise ValueError("zone_low must be lower than zone_high")
        return self


class Decider:
    """Asks the decision model inside the uncertainty zone of layer 2 and returns P(COMPLEX)."""

    def __init__(self, params: DeciderParams, ask: AskDecisionModel) -> None:
        self.params = params
        self._ask = ask

    @classmethod
    def load(cls, path: str | Path, ask: AskDecisionModel) -> "Decider":
        """Load the parameters from a JSON file."""
        return cls(DeciderParams.model_validate(json.loads(Path(path).read_text(encoding="utf-8"))), ask)

    def in_zone(self, knn_score: float) -> bool:
        """Whether layer 2 is uncertain enough to ask the decision model."""
        return self.params.zone_low <= knn_score < self.params.zone_high

    async def score(self, prompt: str, knn_score: float) -> float:
        """P(COMPLEX) from the kNN score and the answers of the decision model; raises on timeout or a missing answer."""
        p = self.params
        answers = await asyncio.wait_for(self._ask(p.model, prompt, p.questions), timeout=p.timeout_seconds)
        z = p.intercept + p.knn_weight * _logit(knn_score) + sum(w * _logit(answers[name]) for name, w in p.weights.items())
        return 1 / (1 + math.exp(-z))
