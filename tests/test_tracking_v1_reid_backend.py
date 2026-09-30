from __future__ import annotations

import numpy as np
import pytest

from tracking.appearance.reid import (
    HSVHistogramReIDBackend,
    cosine_similarity,
    create_reid_backend,
    embed_validated,
    l2_normalize_vector,
    quality_weighted_mean,
    validate_embedding_batch,
)


def test_l2_normalize_vector():
    result = l2_normalize_vector([3.0, 4.0])
    assert result.tolist() == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0, 0.0],
        [1.0, float("nan")],
    ],
)
def test_l2_normalize_rejects_invalid(values):
    with pytest.raises(ValueError):
        l2_normalize_vector(values)


def test_validate_embedding_batch_normalizes_rows():
    result = validate_embedding_batch(
        np.asarray(
            [[3.0, 4.0], [0.0, 2.0]],
            dtype=np.float32,
        ),
        expected_count=2,
        expected_dim=2,
    )
    assert np.linalg.norm(result, axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_validate_embedding_batch_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_embedding_batch(
            np.ones((2, 3), dtype=np.float32),
            expected_count=2,
            expected_dim=4,
        )


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_quality_weighted_mean_biases_toward_high_quality():
    result = quality_weighted_mean(
        ([1.0, 0.0], [0.0, 1.0]),
        (0.9, 0.1),
    )
    assert result[0] > result[1]
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_quality_weighted_mean_validates_lengths():
    with pytest.raises(ValueError):
        quality_weighted_mean(
            ([1.0, 0.0],),
            (0.5, 0.5),
        )


def solid_bgr(color, height=100, width=50):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def test_hsv_backend_has_fixed_dimension_and_normalized_rows():
    backend = HSVHistogramReIDBackend()
    result = embed_validated(
        backend,
        (
            solid_bgr((0, 0, 255)),
            solid_bgr((255, 0, 0)),
        ),
    )

    assert result.shape == (2, 128)
    assert np.linalg.norm(result, axis=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_hsv_backend_same_color_is_more_similar_than_different_color():
    backend = HSVHistogramReIDBackend()
    matrix = embed_validated(
        backend,
        (
            solid_bgr((0, 0, 255)),
            solid_bgr((0, 0, 240)),
            solid_bgr((255, 0, 0)),
        ),
    )

    same = cosine_similarity(matrix[0], matrix[1])
    different = cosine_similarity(matrix[0], matrix[2])

    assert same > different


def test_create_backend_registry():
    backend = create_reid_backend("hsv-hist")
    assert backend.model_id == "hsv-torso-hist-v1"

    with pytest.raises(ValueError, match="unknown"):
        create_reid_backend("not-a-model")
