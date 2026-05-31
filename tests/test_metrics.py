from ccds.metrics import classification_metrics


def test_classification_metrics_shape_and_values():
    metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], num_classes=2)

    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert len(metrics["per_class_accuracy"]) == 2
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2
