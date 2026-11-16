import json
from dataclasses import replace

import pytest
from PIL import Image

from online_ml.data import clean_image, load_manifest, make_split, pixel_hash, save_manifest
from online_ml.encoder import EmbeddingCache
from online_ml.prepare import import_folders


def test_manifest_round_trip_and_relative_paths(records, tmp_path):
    path = tmp_path / "manifests" / "images.jsonl"
    save_manifest(path, records)
    raw = json.loads(path.read_text().splitlines()[0])
    assert raw["path"].startswith("../")
    assert load_manifest(path) == records


def test_duplicate_decoded_pixels_rejected_even_with_different_file_metadata(records, tmp_path):
    duplicate = tmp_path / "duplicate.bmp"
    clean_image(records[0].path).save(duplicate)
    rows = [records[0], replace(
        records[0], id="other-id", path=str(duplicate), sha256="", split="pool"
    )]
    path = tmp_path / "duplicate.jsonl"
    save_manifest(path, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(path)


def test_changed_image_rejected(records, tmp_path):
    path = tmp_path / "manifest.jsonl"
    save_manifest(path, records)
    Image.new("RGB", (12, 12), "red").save(records[0].path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_manifest(path)


def test_unreviewed_evaluation_rejected(records, tmp_path):
    path = tmp_path / "manifest.jsonl"
    save_manifest(path, [replace(records[0], reviewed=False)])
    with pytest.raises(ValueError, match="manually reviewed"):
        load_manifest(path)


def test_split_is_deterministic_balanced_and_reviewed(records):
    rows = [replace(row, split="pool") for row in records]
    first = make_split(rows, 3, 42)
    assert first == make_split(list(reversed(rows)), 3, 42)[::-1]
    held_out = [r for r in first if r.split == "eval"]
    assert len(held_out) == 6
    assert sum(r.label for r in held_out) == 3
    assert all(r.reviewed for r in held_out)
    with pytest.raises(ValueError, match="already has"):
        make_split(first, 3, 42)


def test_split_requires_enough_reviewed_images(records):
    with pytest.raises(ValueError, match="need at least"):
        make_split([replace(r, split="pool") for r in records], 7, 42)


def test_uncertain_split_holds_out_scores_nearest_the_boundary(records):
    rows = [replace(row, split="pool") for row in records if row.reviewed]
    scores = {(r.preset, r.id): 0.5 + (0.01 * n if r.label else 0.4 - 0.01 * n)
              for n, r in enumerate(rows)}
    held_out = [r for r in make_split(rows, 2, 42, scores) if r.split == "eval"]
    for label in (0, 1):
        group = sorted((r for r in rows if r.label == label),
                       key=lambda r: abs(scores[(r.preset, r.id)] - 0.5))
        assert {r.id for r in held_out if r.label == label} == {r.id for r in group[:2]}


def test_shared_split_alternates_borderline_photos_between_sets(records):
    rows = [replace(row, split="pool") for row in records if row.reviewed]
    scores = {(r.preset, r.id): 0.5 + 0.01 * n * (1 if r.label else -1)
              for n, r in enumerate(rows)}
    split = make_split(rows, 2, 42, scores, share=True)
    for label in (0, 1):
        ranked = sorted((r for r in split if r.label == label),
                        key=lambda r: abs(scores[(r.preset, r.id)] - 0.5))
        assert [r.split for r in ranked[:4]] == ["eval", "pool", "eval", "pool"]


def test_shared_split_tops_up_small_classes_from_the_easiest_photos(records):
    rows = [replace(row, split="pool") for row in records if row.reviewed]
    scores = {(r.preset, r.id): 0.5 + 0.01 * n * (1 if r.label else -1)
              for n, r in enumerate(rows)}
    split = make_split(rows, 5, 42, scores, share=True)
    for label in (0, 1):
        ranked = sorted((r for r in split if r.label == label),
                        key=lambda r: abs(scores[(r.preset, r.id)] - 0.5))
        assert sum(r.split == "eval" for r in ranked) == 5
        assert ranked[1].split == "pool"  # the second-hardest photo stays labelable


def test_exif_removed_and_decoded_identity_stable():
    image = Image.new("RGB", (8, 8), "blue")
    image.getexif()[270] = "private description"
    clean = clean_image(image)
    assert not clean.getexif()
    assert not clean.info
    assert pixel_hash(image) == pixel_hash(clean)


def test_file_hash_memo_is_invalidated_when_the_file_changes(tmp_path):
    import os

    path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    first = pixel_hash(path)
    assert pixel_hash(path) == first == pixel_hash(Image.open(path))
    Image.new("RGB", (8, 8), "red").save(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert pixel_hash(path) != first
    assert pixel_hash(path) == pixel_hash(Image.open(path))


def test_embedding_cache_is_versioned_atomic_and_rejects_bad_data(tmp_path):
    import numpy as np

    first = EmbeddingCache(tmp_path, {"revision": "a", "preprocessing": 1})
    second = EmbeddingCache(tmp_path, {"revision": "b", "preprocessing": 1})
    first.put("image", np.array([3, 4]))
    np.testing.assert_allclose(first.get("image"), [0.6, 0.8])
    assert second.get("image") is None
    (first.root / "image.npy").write_bytes(b"corrupt")
    assert first.get("image") is None
    assert not list(first.root.glob("tmp*"))


def test_import_requires_explicit_review_confirmation(tmp_path):
    pos, neg = tmp_path / "pos", tmp_path / "neg"
    pos.mkdir()
    neg.mkdir()
    Image.new("RGB", (8, 8), "red").save(pos / "one.png")
    Image.new("RGB", (8, 8), "blue").save(neg / "two.png")
    output = tmp_path / "import.jsonl"
    rows = import_folders(output, "people", pos, neg)
    assert len(rows) == 2
    assert not any(row.reviewed for row in load_manifest(output))
    with pytest.raises(FileExistsError):
        import_folders(output, "people", pos, neg)
