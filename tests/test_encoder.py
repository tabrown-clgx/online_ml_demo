"""Exercise the actual Transformers feature APIs without downloading model weights."""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from online_ml.encoder import SiglipEncoder  # noqa: E402


@pytest.mark.parametrize("naflex", [False, True])
def test_real_model_feature_apis_and_image_cache(tmp_path, naflex):
    # The default fixed-resolution SigLIP2 checkpoint uses the SiglipModel class;
    # NaFlex checkpoints use Siglip2Model. AutoModel must support either family.
    torch.manual_seed(42)
    text = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, max_position_embeddings=8,
                pad_token_id=0, bos_token_id=1, eos_token_id=2)
    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, patch_size=4)
    if naflex:
        model = transformers.Siglip2Model(transformers.Siglip2Config(
            text_config=text, vision_config={**vision, "num_patches": 16},
        ))
        image_processor = transformers.Siglip2ImageProcessor(patch_size=4, max_num_patches=16)
    else:
        model = transformers.SiglipModel(transformers.SiglipConfig(
            text_config=text, vision_config={**vision, "image_size": 16},
        ))
        image_processor = transformers.SiglipImageProcessor(size={"height": 16, "width": 16})

    class Processor:
        tokenizer = SimpleNamespace(init_kwargs={"offline": True})

        def __init__(self):
            self.image_processor = image_processor

        def __call__(self, *, images=None, text=None, **kwargs):
            if images is not None:
                return image_processor(images=images, return_tensors="pt")
            # A deterministic offline tokenizer stand-in keeps tests independent
            # of downloaded vocabulary files, while exercising the real text tower.
            ids = [[b % 31 for b in hashlib.sha256(prompt.encode()).digest()[:8]]
                   for prompt in text]
            return transformers.BatchEncoding({"input_ids": torch.tensor(ids)})

    encoder = SiglipEncoder(cache_dir=tmp_path, device="cpu")
    with (
        patch.object(transformers.AutoModel, "from_pretrained", return_value=model),
        patch.object(transformers.AutoProcessor, "from_pretrained", return_value=Processor()),
        patch.object(model, "get_image_features", wraps=model.get_image_features) as image_call,
    ):
        image = Image.new("RGB", (24, 20), "red")
        values = encoder.encode_images([image])
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1)
        assert values.shape == (1, 16)
        np.testing.assert_array_equal(values, encoder.encode_images([image]))
        assert image_call.call_count == 1
        prior = encoder.text_prior("a person", "an empty scene")
        assert prior.shape == (16,)
        assert np.isfinite(prior).all()
        assert encoder.identity["normalization"] == "l2"
        assert not any(parameter.requires_grad for parameter in model.parameters())
        assert encoder.encode_images([]).shape == (0, 16)
