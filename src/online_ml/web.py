"""Export a static, model-free copy of the demo for GitHub Pages.

The encoder is frozen, so every photo embedding and each task's text prior can
be computed once here. The browser only runs the tiny probe (``docs/probe.js``),
which mirrors ``probe.TextPriorProbe`` exactly.
"""

import base64
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .data import ImageRecord, clean_image, pixel_hash
from .presets import PRESETS

IMAGE_SIZE = 400


def _b64(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def load_open_images_credits(path: str | Path) -> dict[str, dict]:
    """Read Open Images' image metadata CSV (author, original page, license)."""
    credits = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            credits[row["ImageID"]] = {
                "author": row["Author"], "author_url": row["AuthorProfileURL"],
                "url": row["OriginalLandingURL"], "license": row["License"],
                "title": row["Title"],
            }
    return credits


def _credit(row: ImageRecord, credits: dict[str, dict]) -> dict:
    if row.id in credits:
        return credits[row.id]
    if row.source.startswith("open-images"):
        raise ValueError(f"No author credit for Open Images photo {row.id}; pass the "
                         "Open Images metadata CSV with --credits.")
    return {"author": "", "author_url": "", "url": row.source_url, "license": row.license,
            "title": ""}


def export_web(
    encoder, records: list[ImageRecord], output: str | Path, *,
    credits: dict[str, dict] | None = None, regularization: float = 0.1,
    optimizer_steps: int = 300,
) -> dict:
    credits = credits or {}
    output = Path(output)
    for generated in ("data", "img"):
        shutil.rmtree(output / generated, ignore_errors=True)
        (output / generated).mkdir(parents=True)
    tasks = []
    written: set[str] = set()
    for key in [k for k in PRESETS if any(r.preset == k for r in records)]:
        preset = PRESETS[key]
        rows = sorted((r for r in records if r.preset == key and r.reviewed),
                      key=lambda r: r.id)
        if not any(r.split == "eval" for r in rows):
            raise ValueError(f"Task {key} has no held-out photos; split the manifest first.")
        prior = encoder.text_prior(preset.positive, preset.negative)
        embeddings = encoder.encode_images([r.path for r in rows])
        photos = []
        for row, vector in zip(rows, embeddings, strict=True):
            name = pixel_hash(row.path)[:20] + ".jpg"
            if name not in written:
                image = clean_image(row.path)
                image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))
                image.save(output / "img" / name, quality=82, optimize=True)
                written.add(name)
            photos.append({
                "id": row.id, "split": row.split, "label": row.label, "image": f"img/{name}",
                "embedding": _b64(vector), "credit": _credit(row, credits),
            })
        task = {
            "key": key, "title": preset.title,
            "positive_name": preset.positive_name, "negative_name": preset.negative_name,
            "positive": preset.positive, "negative": preset.negative,
            "prior": _b64(prior), "dim": int(len(prior)), "photos": photos,
        }
        (output / "data" / f"{key}.json").write_text(json.dumps(task, separators=(",", ":")))
        tasks.append({k: task[k] for k in ("key", "title", "positive_name", "negative_name")}
                     | {"file": f"data/{key}.json",
                        "held_out": sum(p["split"] == "eval" for p in photos),
                        "pool": sum(p["split"] == "pool" for p in photos)})
    index = {
        "tasks": tasks, "regularization": regularization, "optimizer_steps": optimizer_steps,
        "encoder": {k: encoder.identity.get(k) for k in ("model", "revision")},
    }
    (output / "data" / "index.json").write_text(json.dumps(index, indent=2))
    return index
