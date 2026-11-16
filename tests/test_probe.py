import numpy as np
import pytest

from online_ml.probe import TextPriorProbe, classification_metrics, sigmoid, unit_rows


def test_zero_labels_match_text_baseline():
    x = np.array([[1.0, 2.0], [-3.0, 1.0]])
    prior = np.array([1.5, -0.2])
    probe = TextPriorProbe(prior)
    expected = sigmoid(unit_rows(x) @ prior)
    np.testing.assert_array_equal(probe.scores(x), probe.scores(x, baseline=True))
    np.testing.assert_allclose(probe.scores(x), expected)
    probe.fit(np.empty((0, 2)), np.array([]))
    np.testing.assert_array_equal(probe.scores(x), expected)


def test_one_class_feedback_is_allowed_and_changes_prediction():
    probe = TextPriorProbe(np.zeros(2))
    x = np.array([[1.0, 0.0]])
    probe.fit(x, np.array([1]))
    assert probe.scores(x)[0] > 0.8
    probe.fit(x, np.array([0]))
    assert probe.scores(x)[0] < 0.2


def test_class_balancing_resists_a_lopsided_feedback_set():
    # One positive and nine negatives at the same (uninformative) point: balanced
    # training treats the classes equally; unbalanced training sides with the majority.
    x = np.ones((10, 2))
    y = np.array([1] + [0] * 9)
    balanced = TextPriorProbe(np.zeros(2)).fit(x, y)
    unbalanced = TextPriorProbe(np.zeros(2), balanced=False).fit(x, y)
    assert balanced.scores(x[:1])[0] == pytest.approx(0.5, abs=1e-6)
    assert unbalanced.scores(x[:1])[0] < 0.2


def test_fit_rebuilds_and_does_not_depend_on_previous_updates():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.5, 1.0]])
    y = np.array([1, 0, 1])
    probe = TextPriorProbe(np.array([0.0, 1.0]))
    probe.fit(x, 1 - y)
    probe.fit(x, y)
    reference = TextPriorProbe(probe.prior).fit(x[::-1], y[::-1])
    np.testing.assert_allclose(probe.weights, reference.weights, atol=1e-12)
    assert probe.bias == pytest.approx(reference.bias)


def test_sigmoid_is_numerically_stable():
    np.testing.assert_array_equal(sigmoid(np.array([-10000, 0, 10000])), [0, 0.5, 1])


@pytest.mark.parametrize("values", [np.zeros((1, 2)), np.array([[np.nan, 1]]), np.ones(3)])
def test_bad_embeddings_rejected(values):
    with pytest.raises(ValueError):
        unit_rows(values)


@pytest.mark.parametrize("strength", [0, -1, float("nan"), float("inf")])
def test_bad_regularization_rejected(strength):
    with pytest.raises(ValueError):
        TextPriorProbe(np.ones(2), strength)


def test_metrics_report_both_recalls_and_confusion_counts():
    metric = classification_metrics(np.array([1, 1, 0, 0]), np.array([0.8, 0.2, 0.6, 0.1]))
    assert (metric["tp"], metric["tn"], metric["fp"], metric["fn"]) == (1, 1, 1, 1)
    assert metric["balanced_accuracy"] == 0.5
    assert metric["positive_recall"] == metric["negative_recall"] == 0.5


def test_empty_and_one_class_metrics_do_not_invent_balanced_accuracy():
    assert classification_metrics(np.array([]), np.array([]))["accuracy"] is None
    assert classification_metrics(np.array([1]), np.array([0.9]))["balanced_accuracy"] is None
