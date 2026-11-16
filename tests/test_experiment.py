import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from online_ml.experiment import create_experiment, synthetic_experiment
from online_ml.tracking import comparable_runs, export_csv, read_run


def make_experiment(encoder, records, tmp_path, **kwargs):
    return create_experiment(
        encoder, records, preset="people", positive="people", negative="empty scene",
        runs_dir=tmp_path / "runs", **kwargs,
    )


def test_labels_corrections_removal_and_reset(encoder, records, tmp_path):
    experiment = make_experiment(encoder, records, tmp_path)
    image_id = experiment.ids("pool")[0]
    experiment.label(image_id, 1)
    experiment.label(image_id, 0)
    reference = make_experiment(encoder, records, tmp_path)
    reference.label(image_id, 0)
    np.testing.assert_array_equal(experiment.probe.weights, reference.probe.weights)
    assert experiment.history[-1]["n_labels"] == 1
    experiment.label(image_id, None)
    np.testing.assert_array_equal(experiment.probe.weights, experiment.probe.prior)
    experiment.label(image_id, 1)
    experiment.reset()
    assert not experiment.feedback
    assert experiment.probe.bias == 0
    np.testing.assert_array_equal(experiment.probe.weights, experiment.probe.prior)
    events = [json.loads(line)["event"] for line in experiment.log.path.read_text().splitlines()]
    assert "reset" in events


def test_evaluation_cannot_be_labeled_or_uploaded(encoder, records, tmp_path):
    experiment = make_experiment(encoder, records, tmp_path)
    image_id = experiment.ids("eval")[0]
    with pytest.raises(ValueError, match="Held-out"):
        experiment.label(image_id, 1)
    image = Image.open(experiment.records[image_id].path)
    with pytest.raises(ValueError, match="duplicates a held-out"):
        experiment.add_upload(image, encoder.encode_images([image])[0])
    assert not experiment.feedback


def test_simulation_excludes_unreviewed_and_eval_and_replays(encoder, records, tmp_path):
    first = make_experiment(encoder, records, tmp_path, seed=7)
    second = make_experiment(encoder, records, tmp_path, seed=7)
    first.simulate(100, "random")
    second.simulate(100, "random")
    expected = {r.id for r in records if r.split == "pool" and r.reviewed}
    assert set(first.feedback) == expected
    assert list(first.feedback.items()) == list(second.feedback.items())
    np.testing.assert_array_equal(first.probe.weights, second.probe.weights)


def test_uncertainty_simulation_replays(encoder, records, tmp_path):
    first = make_experiment(encoder, records, tmp_path)
    second = make_experiment(encoder, records, tmp_path)
    first.simulate(5)
    second.simulate(5)
    assert list(first.feedback.items()) == list(second.feedback.items())


def test_upload_deduplicates_pool_and_keeps_paths_out_of_logs(encoder, records, tmp_path):
    experiment = make_experiment(encoder, records, tmp_path)
    existing = records[5]
    image = Image.open(existing.path)
    assert experiment.add_upload(image, encoder.encode_images([image])[0]) == existing.id
    upload = Image.new("RGB", (9, 9), "purple")
    image_id = experiment.add_upload(upload, encoder.encode_images([upload])[0])
    assert experiment.records[image_id].split == "pool"
    assert not experiment.records[image_id].reviewed
    experiment.label(image_id, 1)
    assert str(tmp_path) not in experiment.log.path.read_text()
    assert experiment.history[-1]["probe"]["n"] == 4


def test_export_and_compatible_run_matching(encoder, records, tmp_path):
    first = make_experiment(encoder, records, tmp_path)
    second = make_experiment(encoder, records, tmp_path, regularization=0.1)
    changed = [replace(records[0], label=1 - records[0].label), *records[1:]]
    make_experiment(encoder, changed, tmp_path)
    compatible = comparable_runs(tmp_path / "runs", first.config)
    assert {r[0]["run_id"] for r in compatible} == {first.log.id, second.log.id}
    first.simulate(2)
    config, history = read_run(first.log.path)
    assert config["encoder"] == encoder.identity
    assert len(history) == 3
    csv = export_csv(first.log.path).read_text()
    assert "probe_balanced_accuracy" in csv
    assert len(csv.splitlines()) == 4


def test_numerical_fixture_improves_but_is_marked_synthetic(tmp_path):
    experiment = synthetic_experiment(runs_dir=tmp_path, seed=42)
    experiment.simulate(20)
    metric = experiment.history[-1]
    assert metric["probe"]["balanced_accuracy"] > metric["baseline"]["balanced_accuracy"]
    assert experiment.config["encoder"]["backend"] == "synthetic"


def test_no_manifest_upload_only_mode(encoder, tmp_path):
    experiment = make_experiment(encoder, [], tmp_path)
    assert experiment.history[-1]["probe"]["n"] == 0
    assert experiment.history[-1]["probe"]["balanced_accuracy"] is None
    assert experiment.rank_pool() == []
