"""Optional Open Images download adapter. FiftyOne is not needed to run the demo."""

from pathlib import Path

from .data import ImageRecord, file_hash, fingerprint, pixel_hash, save_manifest
from .presets import PRESETS


def download_candidates(
    output: str | Path, presets: list[str], per_class: int = 80, seed: int = 42,
) -> list[ImageRecord]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    if per_class < 1:
        raise ValueError("Candidate count must be positive.")
    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError('Install the optional downloader: pip install -e ".[data]"') from exc
    records = []
    for key in presets:
        preset = PRESETS[key]
        seen: set[str] = set()
        for label, classes in ((1, preset.positive_classes), (0, preset.negative_classes)):
            print(f"Downloading up to {per_class} {key} candidates, class {label}: {classes}")
            dataset = foz.load_zoo_dataset(
                "open-images-v7", split="validation", label_types=["classifications"],
                classes=list(classes), max_samples=per_class, shuffle=True, seed=seed,
                only_matching=False,
                # The zoo resolves its download directory from FiftyOne config.
                # Passing dataset_dir here also forwards it to the importer,
                # which already receives dataset_dir internally (duplicate kwarg).
                dataset_name="online-ml-" + fingerprint([key, label, per_class, seed])[:16],
            )
            for sample in dataset:
                # The zoo may return fallback samples when class coverage is low.
                # Require explicit positive source annotations, never absent labels.
                positive = sample["positive_labels"]
                source_classes = {c.label for c in positive.classifications} if positive else set()
                matching = source_classes.intersection(classes)
                if not matching:
                    continue
                image_path = Path(sample.filepath).resolve()
                digest = pixel_hash(image_path)
                if digest in seen:
                    continue
                seen.add(digest)
                image_id = image_path.stem
                records.append(ImageRecord(
                    id=image_id, path=str(image_path), preset=key, label=label,
                    reviewed=False, source="open-images-v7/validation",
                    source_url=("https://storage.googleapis.com/openimages/web/visualizer/"
                                f"index.html?set=validation&type=classification&image_id={image_id}"),
                    license="CC-BY-2.0; verify original image attribution via source_url",
                    source_class=", ".join(sorted(matching)), sha256=file_hash(image_path),
                ))
        print(f"{key}: {sum(r.preset == key for r in records)} unique candidates. {preset.note}")
    if not records:
        raise ValueError("No candidates matched positive source annotations.")
    save_manifest(output, records)
    return records


def import_folders(
    output: str | Path, preset: str, positive: str | Path, negative: str | Path,
    *, reviewed: bool = False, license_text: str = "user-supplied; verify permission",
) -> list[ImageRecord]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}.")
    records = []
    seen = set()
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for label, directory in ((1, positive), (0, negative)):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError(f"Image directory does not exist: {directory}")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            digest = pixel_hash(path)
            if digest in seen:
                raise ValueError(f"Duplicate image in imported folders: {path}")
            seen.add(digest)
            records.append(ImageRecord(
                id=f"local-{digest[:20]}", path=str(path), preset=preset,
                label=label, reviewed=reviewed, sha256=file_hash(path), license=license_text,
            ))
    if not records:
        raise ValueError("No supported image files found.")
    save_manifest(output, records)
    return records
