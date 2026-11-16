"""Lazy SigLIP2 loading and a versioned, content-addressed embedding cache."""

import importlib.metadata
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .data import clean_image, fingerprint, pixel_hash
from .presets import DEFAULT_MODEL, PROMPT_TEMPLATES
from .probe import unit_rows


class EmbeddingCache:
    def __init__(self, root: str | Path, identity: dict):
        self.root = Path(root) / fingerprint(identity)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, digest: str) -> np.ndarray | None:
        path = self.root / f"{digest}.npy"
        if not path.exists():
            return None
        try:
            value = np.load(path, allow_pickle=False)
            if (value.ndim == 1 and np.isfinite(value).all()
                    and np.isclose(np.linalg.norm(value), 1.0, rtol=0, atol=1e-8)):
                # Do not repeatedly normalize a cache hit: preserve exact vectors
                # across precomputation, UI sessions, and deterministic replays.
                return value
        except (OSError, ValueError, TypeError):
            pass
        return None

    def put(self, digest: str, embedding: np.ndarray) -> np.ndarray:
        value = unit_rows(np.asarray(embedding)[None, :])[0]
        fd, temporary = tempfile.mkstemp(suffix=".npy", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as target:
                np.save(target, value, allow_pickle=False)
            os.replace(temporary, self.root / f"{digest}.npy")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return value


class SiglipEncoder:
    def __init__(
        self, model_id: str = DEFAULT_MODEL, *, revision: str = "main",
        device: str = "auto", cache_dir: str | Path = ".cache/embeddings",
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError('Install model dependencies: pip install -e ".[model]"') from exc
        if self.device == "auto":
            self.device = ("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        resolved = getattr(model.config, "_commit_hash", None) or self.revision
        processor = AutoProcessor.from_pretrained(self.model_id, revision=resolved)
        model = model.to(self.device).eval()
        model.requires_grad_(False)
        identity = {
            "backend": "siglip2", "model": self.model_id, "revision": resolved,
            "device": self.device, "dtype": "float32",
            "transformers": importlib.metadata.version("transformers"),
            "torch": str(torch.__version__),
            "numpy": np.__version__, "pillow": importlib.metadata.version("Pillow"),
            "preprocessing": processor.image_processor.to_dict(),
            "tokenizer": processor.tokenizer.init_kwargs,
            "normalization": "l2", "cache_version": 2,
        }
        cache = EmbeddingCache(self.cache_dir, identity)
        # Publish the loaded sentinel last, so a failed load/cache initialization
        # can be retried rather than leaving a partially initialized encoder.
        self._torch = torch
        self._processor = processor
        self._identity = identity
        self._cache = cache
        self._model = model

    @property
    def identity(self) -> dict:
        with self._lock:
            self._load()
            return self._identity.copy()

    def encode_images(self, images: list[str | Path | Image.Image]) -> np.ndarray:
        with self._lock:
            self._load()
            vectors = []
            # Small batches keep memory predictable on CPU. Each cached image is
            # independently addressable, so uploads and precomputation share a cache.
            for start in range(0, len(images), 8):
                batch = images[start:start + 8]
                # Hash first (memoized for files); decode only the cache misses.
                digests = [pixel_hash(image) for image in batch]
                cached = [self._cache.get(digest) for digest in digests]
                missing = [i for i, value in enumerate(cached) if value is None]
                if missing:
                    inputs = self._processor(
                        images=[clean_image(batch[i]) for i in missing], return_tensors="pt"
                    ).to(self.device)
                    with self._torch.inference_mode():
                        features = self._model.get_image_features(**inputs)
                    values = unit_rows(features.detach().float().cpu().numpy())
                    for i, value in zip(missing, values, strict=True):
                        cached[i] = self._cache.put(digests[i], value)
                vectors.extend(cached)
            if not vectors:
                return np.empty((0, self._model.config.vision_config.hidden_size))
            return np.stack(vectors)

    def text_prior(self, positive: str, negative: str) -> np.ndarray:
        if not positive.strip() or not negative.strip() or positive.strip() == negative.strip():
            raise ValueError("Provide different, nonempty positive and negative descriptions.")
        with self._lock:
            self._load()
            prompts = [template.format(text) for text in (positive, negative)
                       for template in PROMPT_TEMPLATES]
            inputs = self._processor(
                text=prompts, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            with self._torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            vectors = unit_rows(features.detach().float().cpu().numpy())
            # Average normalized prompt vectors, rather than renormalizing the
            # mean: x @ prior exactly equals the difference of mean prompt logits.
            positive_vector, negative_vector = vectors.reshape(2, len(PROMPT_TEMPLATES), -1).mean(1)
            scale = self._model.logit_scale.exp().item()
            # SigLIP's shared logit_bias cancels in a two-description softmax.
            return scale * (positive_vector - negative_vector)
