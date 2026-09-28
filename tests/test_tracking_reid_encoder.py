from __future__ import annotations

import numpy as np
import pytest

from tracking.reid_encoder import (
    ReIDEncoderConfig,
    TorchreidEncoder,
    l2_normalize_rows,
)


class FakeExtractor:
    def __init__(self):
        self.received = []

    def __call__(self, images):
        self.received.extend(
            [image.copy() for image in images]
        )
        rows = []
        for index, _ in enumerate(images):
            rows.append(
                [1.0 + index, 2.0, 3.0]
            )
        return np.asarray(rows, dtype=np.float32)


def test_bgr_is_converted_to_rgb_before_backend():
    fake = FakeExtractor()
    encoder = TorchreidEncoder(
        None,
        config=ReIDEncoderConfig(
            batch_size=8,
        ),
        extractor=fake,
    )

    # Pure red in OpenCV BGR.
    image = np.zeros((8, 4, 3), dtype=np.uint8)
    image[:] = (0, 0, 255)

    features = encoder.encode_bgr([image])

    assert features.shape == (1, 3)
    assert fake.received[0][0, 0].tolist() == [255, 0, 0]


def test_embeddings_are_l2_normalized():
    fake = FakeExtractor()
    encoder = TorchreidEncoder(
        None,
        extractor=fake,
    )

    images = [
        np.zeros((16, 8, 3), dtype=np.uint8),
        np.ones((16, 8, 3), dtype=np.uint8),
    ]

    features = encoder.encode_bgr(images)

    norms = np.linalg.norm(
        features,
        axis=1,
    )
    assert np.allclose(norms, 1.0)


def test_batching_preserves_number_of_rows():
    fake = FakeExtractor()
    encoder = TorchreidEncoder(
        None,
        config=ReIDEncoderConfig(
            batch_size=2,
        ),
        extractor=fake,
    )

    images = [
        np.zeros((16, 8, 3), dtype=np.uint8)
        for _ in range(5)
    ]

    features = encoder.encode_bgr(images)

    assert features.shape == (5, 3)
    assert len(fake.received) == 5


def test_zero_norm_features_are_rejected():
    with pytest.raises(ValueError):
        l2_normalize_rows(
            np.zeros((2, 4), dtype=np.float32)
        )


def test_invalid_image_shape_is_rejected():
    encoder = TorchreidEncoder(
        None,
        extractor=FakeExtractor(),
    )

    with pytest.raises(ValueError):
        encoder.encode_bgr(
            [
                np.zeros(
                    (16, 8),
                    dtype=np.uint8,
                )
            ]
        )
