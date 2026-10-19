# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Layer 1: text heuristics that award points for signs of a complex prompt (keyword lists, length, digits, URLs)."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LengthStep(BaseModel):
    chars: int = Field(ge=1)
    points: float


class DigitRule(BaseModel):
    over: float = Field(ge=0.0, le=1.0)
    points: float


class HeuristicPoints(BaseModel):
    """How many points each sign is worth; positive points push towards the large model."""

    complex_phrase: float = 1.0
    simple_phrase: float = -1.0
    url: float = 1.0
    length: list[LengthStep] = [LengthStep(chars=400, points=1.0), LengthStep(chars=1500, points=1.0)]
    digits: DigitRule | None = DigitRule(over=0.1, points=-1.0)


def _phrases(values: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(rf"\b{re.escape(value.lower())}\b") for value in values]


class HeuristicScorer:
    """Sums the points of every sign found in the prompt; the larger the sum, the more likely the prompt is complex."""

    def __init__(self, complex_phrases: Sequence[str], simple_phrases: Sequence[str], points: HeuristicPoints | None = None) -> None:
        self._complex = _phrases(complex_phrases)
        self._simple = _phrases(simple_phrases)
        self._points = points or HeuristicPoints()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeuristicScorer":
        """Build the scorer from parameters (keys: complex_phrases, simple_phrases, optional points)."""
        points = HeuristicPoints.model_validate(data["points"]) if "points" in data else None
        return cls(data["complex_phrases"], data["simple_phrases"], points)

    @classmethod
    def load(cls, path: str | Path) -> "HeuristicScorer":
        """Load the parameters from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def points(self, text: str) -> float:
        """Total points of the prompt."""
        low = text.lower()
        rules = self._points
        total = rules.complex_phrase * sum(bool(p.search(low)) for p in self._complex)
        total += rules.simple_phrase * sum(bool(p.search(low)) for p in self._simple)
        total += sum(step.points for step in rules.length if len(text) > step.chars)
        if rules.digits is not None and sum(c.isdigit() for c in text) / max(len(text), 1) > rules.digits.over:
            total += rules.digits.points
        if "http" in low:
            total += rules.url
        return total
