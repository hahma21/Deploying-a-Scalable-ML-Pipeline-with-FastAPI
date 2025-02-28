import pytest
# TODO: add necessary import

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    # add description for the first test
    """
    # Your code here
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output():
    """
    # add description for the second test
    """
    # Your code here
    model = RandomForestClassifier()
    X_test = np.random.rand(10, 5)
    model.fit(X_test, np.random.randint(0, 2, 10))
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Inference output is not a numpy array"
    assert len(preds) == len(X_test), "Inference output length mismatch"


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # add description for the third test
    """
    # Your code here
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    
    assert 0 <= precision <= 1, "Precision is out of bounds"
    assert 0 <= recall <= 1, "Recall is out of bounds"
    assert 0 <= fbeta <= 1, "F1-score is out of bounds"
