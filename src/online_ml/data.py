"""JSONL manifests, reviewed splits, and content-based leakage checks."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .presets import PRESETS


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_image(path: str | Path | Image.Image) -> Image.Image:
    """Honor orientation, then discard EXIF and other metadata. Never rewrite originals."""
    if isinstance(path, Image.Image):
        image = ImageOps.exif_transpose(path).convert("RGB")
    else:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    return Image.frombytes("RGB", image.size, image.tobytes())


_PATH_HASHES: dict[tuple[str, int, int], str] = {}


def pixel_hash(path: str | Path | Image.Image) -> str:
    """Content identity of the decoded, orientation-corrected pixels. File results are
    memoized by (path, mtime, size), so repeated lookups skip decoding the image."""
    key = None
    if not isinstance(path, Image.Image):
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = (str(resolved), stat.st_mtime_ns, stat.st_size)
        if key in _PATH_HASHES:
            return _PATH_HASHES[key]
    image = clean_image(path)
    h = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
    h.update(image.tobytes())
    digest = h.hexdigest()
    if key is not None:
        _PATH_HASHES[key] = digest
    return digest


@dataclass(frozen=True)
class ImageRecord:
    id: str
    path: str
    preset: str
    label: int
    split: str = "pool"
    reviewed: bool = False
    source: str = "local"
    source_url: str = ""
    license: str = "user-supplied; verify permission"
    source_class: str = ""
    sha256: str = ""


def load_manifest(path: str | Path) -> list[ImageRecord]:
    path = Path(path).resolve()
    rows = []
    seen_ids: set[tuple[str, str]] = set()
    # The same image may be used in several binary tasks, but never twice in one task.
    seen_pixels: set[tuple[str, str]] = set()
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = ImageRecord(**json.loads(line))
                if not isinstance(row.id, str) or not row.id:
                    raise ValueError("Image ID must be a nonempty string.")
                if not isinstance(row.path, str) or not row.path:
                    raise ValueError("Image path must be a nonempty string.")
                resolved = (path.parent / row.path).resolve()
                row = replace(row, path=str(resolved))
                if row.preset not in PRESETS or row.split not in {"pool", "eval"}:
                    raise ValueError("Unknown preset or split.")
                if type(row.label) is not int or row.label not in (0, 1):
                    raise ValueError("Label must be integer 0 or 1.")
                if type(row.reviewed) is not bool or not row.id:
                    raise ValueError("Expected a review boolean and a nonempty ID.")
                if row.split == "eval" and not row.reviewed:
                    raise ValueError("Evaluation labels must be manually reviewed.")
                digest = file_hash(resolved)
                if row.sha256 and row.sha256 != digest:
                    raise ValueError("Image changed since manifest creation (SHA-256 mismatch).")
                row = replace(row, sha256=digest)
                identity = (row.preset, row.id)
                content = (row.preset, pixel_hash(resolved))
                if identity in seen_ids or content in seen_pixels:
                    raise ValueError("Duplicate ID or image content within a task; leakage risk.")
                seen_ids.add(identity)
                seen_pixels.add(content)
                rows.append(row)
            except (TypeError, ValueError, OSError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def save_manifest(path: str | Path, records: list[ImageRecord]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also makes review clicks safe against interrupted writes.
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as target:
            for record in records:
                row = asdict(record)
                row["path"] = os.path.relpath(Path(record.path).resolve(), path.parent)
                target.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_split(
    records: list[ImageRecord], eval_per_class: int, seed: int,
    scores: dict[tuple[str, str], float] | None = None, *, share: bool = False,
) -> list[ImageRecord]:
    """Hold out images per preset and class: randomly, or (with zero-shot positive
    scores keyed by (preset, id)) the images closest to the 0.5 decision boundary.
    With ``share``, the hardest images alternate between held-out and the labeling
    pool, so active learning still has borderline photos to ask about."""
    if eval_per_class < 1:
        raise ValueError("Evaluation images per class must be positive.")
    if any(row.split == "eval" for row in records):
        raise ValueError("This manifest already has a fixed evaluation split. Create a new copy.")
    selected: set[tuple[str, str]] = set()
    rng = np.random.default_rng(seed)
    for preset in sorted({row.preset for row in records}):
        for label in (0, 1):
            candidates = sorted(
                r.id for r in records if r.preset == preset and r.label == label and r.reviewed
            )
            if len(candidates) <= eval_per_class:
                raise ValueError(
                    f"{preset}, class {label}: need at least {eval_per_class + 1} reviewed "
                    "images to keep evaluation and feedback examples."
                )
            if scores is None:
                chosen = rng.choice(candidates, eval_per_class, replace=False)
            else:
                ranked = sorted(candidates, key=lambda i: (abs(scores[(preset, i)] - 0.5), i))
                if share:
                    chosen = ranked[::2][:eval_per_class]
                    # Small classes: top up from the easiest photos, keeping the
                    # borderline ones in the labeling pool.
                    rest = [i for i in reversed(ranked) if i not in chosen]
                    chosen += rest[:eval_per_class - len(chosen)]
                else:
                    chosen = ranked[:eval_per_class]
            selected.update((preset, str(i)) for i in chosen)
    return [replace(r, split="eval" if (r.preset, r.id) in selected else "pool") for r in records]
