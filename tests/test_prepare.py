import sys
from types import ModuleType, SimpleNamespace

from PIL import Image

from online_ml.data import load_manifest
from online_ml.prepare import download_candidates


def test_downloader_filters_fallbacks_deduplicates_and_requires_review(tmp_path, monkeypatch):
    class Sample:
        def __init__(self, path, *labels):
            self.filepath = str(path)
            self.labels = SimpleNamespace(
                classifications=[SimpleNamespace(label=label) for label in labels]
            )

        def __getitem__(self, key):
            assert key == "positive_labels"
            return self.labels

    paths = []
    for index, color in enumerate(("red", "blue", "green")):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (8, 8), color).save(path)
        paths.append(path)
    calls = []

    def load(name, **kwargs):
        calls.append(kwargs)
        assert name == "open-images-v7"
        assert kwargs["label_types"] == ["classifications"]
        assert kwargs["max_samples"] == 5
        # FiftyOne forwards extra kwargs to its importer, which already supplies
        # dataset_dir explicitly. This must use the configured zoo cache instead.
        assert "dataset_dir" not in kwargs
        if kwargs["classes"] == ["Person"]:
            return [Sample(paths[0], "Person"), Sample(paths[2], "Animal")]
        # The red image already exists in the positive candidate group. Exclude
        # that duplicate, and never infer task negatives from an absent Person tag.
        return [Sample(paths[0], "Chair"), Sample(paths[1], "Tree")]

    fiftyone, zoo = ModuleType("fiftyone"), ModuleType("fiftyone.zoo")
    zoo.load_zoo_dataset = load
    fiftyone.zoo = zoo
    monkeypatch.setitem(sys.modules, "fiftyone", fiftyone)
    monkeypatch.setitem(sys.modules, "fiftyone.zoo", zoo)
    output = tmp_path / "candidates.jsonl"
    rows = download_candidates(output, ["people"], per_class=5)
    assert len(calls) == 2
    assert len(rows) == 2
    assert {row.label for row in rows} == {0, 1}
    assert all(not row.reviewed and row.split == "pool" for row in load_manifest(output))
    assert all(row.source_url and row.license and row.sha256 for row in rows)
