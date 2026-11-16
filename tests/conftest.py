from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from online_ml.data import ImageRecord, clean_image, file_hash
from online_ml.probe import unit_rows


class FakeEncoder:
    """Offline test encoder. Never exposed as real model performance."""

    identity = {"backend": "test-only", "revision": "1"}

    def encode_images(self, images):
        values = []
        for source in images:
            image = clean_image(source)
            r, g, b = image.getpixel((0, 0))
            values.append([r - 128, g - 128, b + 1])
        return unit_rows(np.array(values).reshape(-1, 3))

    def text_prior(self, positive, negative):
        if not positive.strip() or positive == negative:
            raise ValueError("Invalid prompts")
        return np.array([1.0, -0.5, 0.1])


@pytest.fixture
def encoder():
    return FakeEncoder()


@pytest.fixture
def records(tmp_path: Path):
    result = []
    for i in range(16):
        path = tmp_path / f"image-{i}.png"
        Image.new("RGB", (12, 12), (i * 15, 20 + i, 50)).save(path)
        result.append(ImageRecord(
            id=f"image-{i:02d}", path=str(path), preset="people", label=i % 2,
            split="eval" if i < 4 else "pool", reviewed=i < 14, sha256=file_hash(path),
        ))
    return result
