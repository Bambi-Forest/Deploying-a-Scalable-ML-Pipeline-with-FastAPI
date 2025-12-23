import numpy as np


from ml.model import train_model, compute_model_metrics


# TODO: implement the first test. Change the function name and input as needed


def test_train_model_returns_model():

    """
    Test that train_model returns a trained model object.
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")


def test_model_is_random_forest():

    """
    Test that the trained model is a RandomForestClassifier.
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])

    model = train_model(X, y)
    assert model.__class__.__name__ == "RandomForestClassifier"


def test_compute_model_metrics_output():

    """
    Test that compute_model_metrics returns precision, recall, and fbeta
    within valid ranges.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
