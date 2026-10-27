# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from server.models.api import SystemOneNoulQuestion
from server.router.config import RouterOptions
from server.router.decider import Decider, DeciderParams
from server.router.engine import RouterEngine
from server.router.index import KnnIndex


def _params(**kwargs: Any) -> DeciderParams:
    data: dict[str, Any] = {
        "model": "decider",
        "questions": {"hard": {"type": "noul", "instructions": "Is it hard?"}},
        "intercept": 0.0,
        "knn_weight": 1.0,
        "weights": {"hard": 1.0},
        "zone_low": 0.4,
        "zone_high": 0.6,
    }
    return DeciderParams.model_validate(data | kwargs)


def _answering(p_yes: float, calls: list[str] | None = None) -> Any:
    async def ask(model: str, prompt: str, questions: dict[str, SystemOneNoulQuestion]) -> dict[str, float]:
        if calls is not None:
            calls.append(f"{model}:{prompt}")
        return dict.fromkeys(questions, p_yes)

    return ask


def _index() -> KnnIndex:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    return KnnIndex(vectors, np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))


def _options() -> RouterOptions:
    return RouterOptions.model_validate({"small_model": "small", "large_model": "large", "index_path": "x.npz", "k": 2, "threshold": 0.5})


async def _embed_uncertain(_: str) -> list[float]:
    return [1.0, 1.0]


async def _embed_simple(_: str) -> list[float]:
    return [1.0, 0.0]


def test_params_validation() -> None:
    with pytest.raises(ValidationError, match="one entry per question"):
        _params(weights={"other": 1.0})
    with pytest.raises(ValidationError, match="zone_low must be lower"):
        _params(zone_low=0.5, zone_high=0.5)


@pytest.mark.asyncio
async def test_score_is_logistic_regression_of_logits() -> None:
    decider = Decider(_params(intercept=-0.5, knn_weight=2.0, weights={"hard": 0.5}), _answering(0.8))
    expected = 1 / (1 + math.exp(-(-0.5 + 2.0 * math.log(0.3 / 0.7) + 0.5 * math.log(0.8 / 0.2))))
    assert await decider.score("x", 0.3) == pytest.approx(expected)


@pytest.mark.asyncio
async def test_score_clips_certain_answers() -> None:
    decider = Decider(_params(knn_weight=0.0), _answering(1.0))
    assert 0.0 < await decider.score("x", 0.5) < 1.0


def test_zone_is_half_open() -> None:
    decider = Decider(_params(), _answering(0.5))
    assert decider.in_zone(0.4)
    assert decider.in_zone(0.59)
    assert not decider.in_zone(0.6)
    assert not decider.in_zone(0.39)


def test_load(tmp_path: Path) -> None:
    path = tmp_path / "l3.json"
    path.write_text(_params().model_dump_json(), encoding="utf-8")
    assert Decider.load(path, _answering(0.5)).params.model == "decider"
    path.write_text(json.dumps({"model": "decider"}), encoding="utf-8")
    with pytest.raises(ValidationError):
        Decider.load(path, _answering(0.5))


@pytest.mark.asyncio
async def test_engine_asks_decider_only_in_the_zone() -> None:
    calls: list[str] = []
    decider = Decider(_params(), _answering(0.9, calls))
    uncertain = await RouterEngine(_options(), _index(), _embed_uncertain, decider=decider).decide("tricky")
    simple = await RouterEngine(_options(), _index(), _embed_simple, decider=decider).decide("easy")
    assert (uncertain.model, uncertain.layer) == ("large", "decider")
    assert uncertain.score == pytest.approx(0.5, abs=1e-3)
    assert uncertain.decider_score is not None
    assert uncertain.decider_score > 0.5
    assert (simple.model, simple.layer, simple.decider_score) == ("small", "knn", None)
    assert calls == ["decider:tricky"]


@pytest.mark.asyncio
async def test_engine_decider_can_send_to_small() -> None:
    decision = await RouterEngine(_options(), _index(), _embed_uncertain, decider=Decider(_params(), _answering(0.1))).decide("x")
    assert (decision.model, decision.layer) == ("small", "decider")


@pytest.mark.asyncio
async def test_engine_falls_back_to_large_when_decider_fails() -> None:
    async def broken(model: str, prompt: str, questions: dict[str, SystemOneNoulQuestion]) -> dict[str, float]:
        raise RuntimeError("decision model down")

    decision = await RouterEngine(_options(), _index(), _embed_uncertain, decider=Decider(_params(), broken)).decide("x")
    assert (decision.model, decision.layer) == ("large", "fallback")
    assert "decision model down" in decision.reason
    assert decision.score == pytest.approx(0.5, abs=1e-3)


@pytest.mark.asyncio
async def test_engine_falls_back_to_large_when_decider_times_out() -> None:
    async def slow(model: str, prompt: str, questions: dict[str, SystemOneNoulQuestion]) -> dict[str, float]:
        await asyncio.sleep(1)
        return dict.fromkeys(questions, 0.1)

    decider = Decider(_params(timeout_seconds=0.01), slow)
    decision = await RouterEngine(_options(), _index(), _embed_uncertain, decider=decider).decide("x")
    assert (decision.model, decision.layer) == ("large", "fallback")
    assert "TimeoutError" in decision.reason
