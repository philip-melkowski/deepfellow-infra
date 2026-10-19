# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from pathlib import Path

import numpy as np
import pytest

from server.models.api import AssistantMessage, SystemMessage, TextContentPart, UserMessage
from server.router.config import RouterOptions, RouterRules
from server.router.engine import RouterEngine
from server.router.heuristics import HeuristicScorer
from server.router.index import KnnIndex
from server.router.prompt import last_user_text


def _index() -> KnnIndex:
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    labels = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    return KnnIndex(vectors, labels)


def _options(**kwargs: object) -> RouterOptions:
    return RouterOptions.model_validate({"small_model": "small", "large_model": "large", "index_path": "x.npz", "k": 2, **kwargs})


def test_index_score_follows_neighbours() -> None:
    index = _index()
    assert index.score([1.0, 0.05], 2) == pytest.approx(0.0)
    assert index.score([0.05, 1.0], 2) == pytest.approx(1.0)
    assert 0.0 < index.score([1.0, 1.0], 4) < 1.0


def test_index_k_larger_than_size_is_clamped() -> None:
    assert 0.0 <= _index().score([1.0, 0.0], 100) <= 1.0


def test_index_rejects_wrong_dimension_and_zero_vector() -> None:
    index = _index()
    with pytest.raises(ValueError, match="dimension"):
        index.score([1.0, 0.0, 0.0], 2)
    with pytest.raises(ValueError, match="zero"):
        index.score([0.0, 0.0], 2)


def test_index_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    np.savez(path, vectors=np.eye(3, dtype=np.float32), labels=np.array([0.0, 1.0, 1.0], dtype=np.float32))
    index = KnnIndex.load(path)
    assert index.size == 3
    assert index.dim == 3


@pytest.mark.asyncio
async def test_engine_routes_by_score() -> None:
    async def embed_simple(_: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_complex(_: str) -> list[float]:
        return [0.0, 1.0]

    simple = await RouterEngine(_options(), _index(), embed_simple).decide("what is 2+2")
    complex_ = await RouterEngine(_options(), _index(), embed_complex).decide("prove the theorem")
    assert (simple.model, simple.layer) == ("small", "knn")
    assert (complex_.model, complex_.layer) == ("large", "knn")
    assert complex_.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_engine_threshold_boundary_goes_to_large() -> None:
    async def embed(_: str) -> list[float]:
        return [1.0, 1.0]

    decision = await RouterEngine(_options(threshold=0.5), _index(), embed).decide("x")
    assert decision.score == pytest.approx(0.5, abs=1e-3)
    assert decision.model == ("large" if decision.score is not None and decision.score >= 0.5 else "small")


@pytest.mark.asyncio
async def test_engine_adds_prefix() -> None:
    seen: list[str] = []

    async def embed(text: str) -> list[float]:
        seen.append(text)
        return [1.0, 0.0]

    await RouterEngine(_options(), _index(), embed).decide("hello")
    assert seen == ["classification: hello"]


@pytest.mark.asyncio
async def test_engine_falls_back_to_large_on_embedding_error() -> None:
    async def embed(_: str) -> list[float]:
        raise RuntimeError("embedder down")

    decision = await RouterEngine(_options(), _index(), embed).decide("hello")
    assert (decision.model, decision.layer) == ("large", "fallback")
    assert "embedder down" in decision.reason


@pytest.mark.asyncio
async def test_engine_falls_back_when_dimension_mismatch() -> None:
    async def embed(_: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    decision = await RouterEngine(_options(), _index(), embed).decide("hello")
    assert decision.layer == "fallback"


@pytest.mark.asyncio
async def test_rules_skip_embedding() -> None:
    called = False

    async def embed(_: str) -> list[float]:
        nonlocal called
        called = True
        return [1.0, 0.0]

    options = _options(rules=RouterRules(force_large_over_chars=10, force_large_keywords=["Prove"]))
    long_ = await RouterEngine(options, _index(), embed).decide("x" * 11)
    keyword = await RouterEngine(options, _index(), embed).decide("PROVE")
    assert (long_.model, long_.layer, keyword.model, keyword.layer) == ("large", "rules", "large", "rules")
    assert not called


def test_options_validation() -> None:
    with pytest.raises(ValueError, match="threshold"):
        _options(threshold=1.5)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        _options(k=0)


def test_last_user_text() -> None:
    assert (
        last_user_text([SystemMessage(content="s"), UserMessage(content="a"), AssistantMessage(content="b"), UserMessage(content="c")])
        == "c"
    )
    assert last_user_text([UserMessage(content=[TextContentPart(text="x"), TextContentPart(text="y")])]) == "x\ny"
    assert last_user_text([SystemMessage(content="only system")]) == ""


def test_index_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        KnnIndex(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="one value per vector"):
        KnnIndex(np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32))


@pytest.mark.asyncio
async def test_rules_without_match_fall_through_to_knn() -> None:
    async def embed(_: str) -> list[float]:
        return [1.0, 0.0]

    options = _options(rules=RouterRules(force_large_over_chars=100, force_large_keywords=["prove"]))
    decision = await RouterEngine(options, _index(), embed).decide("hello")
    assert (decision.model, decision.layer) == ("small", "knn")


def _scorer(**points: object) -> HeuristicScorer:
    data: dict[str, object] = {"complex_phrases": ["prove", "step by step"], "simple_phrases": ["what is"]}
    if points:
        data["points"] = points
    return HeuristicScorer.from_dict(data)


def test_heuristic_points_for_phrases() -> None:
    scorer = _scorer()
    assert scorer.points("Prove it step by step") == 2.0
    assert scorer.points("What is a cat") == -1.0
    assert scorer.points("a cat") == 0.0
    assert scorer.points("provement") == 0.0


def test_heuristic_points_for_length_digits_and_url() -> None:
    scorer = _scorer()
    assert scorer.points("x" * 401) == 1.0
    assert scorer.points("x" * 1501) == 2.0
    assert scorer.points("12345 ab") == -1.0
    assert scorer.points("see http://a.b") == 1.0
    assert scorer.points("") == 0.0


def test_heuristic_points_are_configurable() -> None:
    scorer = _scorer(complex_phrase=3.0, simple_phrase=0.0, url=0.0, length=[], digits=None)
    assert scorer.points("prove 123456 http://a.b " + "x" * 2000) == 3.0


def test_heuristics_load_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "l1.json"
    path.write_text(json.dumps({"complex_phrases": ["prove"], "simple_phrases": []}), encoding="utf-8")
    assert HeuristicScorer.load(path).points("prove") == 1.0
    with pytest.raises(KeyError):
        HeuristicScorer.from_dict({"simple_phrases": []})


@pytest.mark.asyncio
async def test_heuristics_over_threshold_send_to_large_without_embedding() -> None:
    async def embed(_: str) -> list[float]:
        raise AssertionError("embedding must not be computed")

    engine = RouterEngine(_options(heuristic_threshold=2.0), _index(), embed, _scorer())
    decision = await engine.decide("Prove it step by step")
    assert decision.layer == "heuristics"
    assert decision.model == "large"
    assert decision.heuristic == 2.0
    assert decision.score is None


@pytest.mark.asyncio
async def test_heuristics_below_threshold_fall_through_to_knn() -> None:
    async def embed(_: str) -> list[float]:
        return [1.0, 0.0]

    engine = RouterEngine(_options(heuristic_threshold=2.0), _index(), embed, _scorer())
    decision = await engine.decide("prove it")
    assert decision.layer == "knn"
    assert decision.model == "small"
    assert decision.heuristic == 1.0


@pytest.mark.asyncio
async def test_engine_without_heuristics_reports_no_points() -> None:
    async def embed(_: str) -> list[float]:
        return [1.0, 0.0]

    decision = await RouterEngine(_options(), _index(), embed).decide("hello")
    assert decision.layer == "knn"
    assert decision.heuristic is None


@pytest.mark.asyncio
async def test_engine_falls_back_when_heuristics_fail() -> None:
    class Broken(HeuristicScorer):
        def points(self, text: str) -> float:
            raise RuntimeError(text)

    async def embed(_: str) -> list[float]:
        return [1.0, 0.0]

    broken = Broken.from_dict({"complex_phrases": [], "simple_phrases": []})
    decision = await RouterEngine(_options(), _index(), embed, broken).decide("hello")
    assert decision.layer == "fallback"
    assert decision.model == "large"
