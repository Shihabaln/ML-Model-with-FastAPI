"""
Model steps module
"""
from numpy import mean, std
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    make_scorer,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def set_up_classifier():
    """
    Set up the classifier for the model training.

    Returns:
    clf (RandomForestClassifier): An instance of RandomForestClassifier.
    """
    return RandomForestClassifier(n_estimators=100)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns the trained classifier.

    Parameters:
    X_train (array): The feature data for training.
    y_train (array): The label data for training.

    Returns:
    clf (RandomForestClassifier): The trained classifier.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    # Set up classifier
    clf = set_up_classifier()

    # Perform cross-validation
    scores = cross_val_score(
        clf, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1
    )

    # Log the cross-validation results
    logger.info("Mean Accuracy: %.3f (%.3f)" % (mean(scores), std(scores)))

    # Train the classifier
    clf.fit(X_train, y_train)

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    logger.info(
        "Precision score: %s\nRecall score: %s\nFbeta score: %s"
        % (precision, recall, fbeta)
    )

    return precision, recall, fbeta


def inference(clf, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if clf is None:
        logger.error("model is None")
        raise ValueError
    y_preds = clf.predict(X)
    logger.info("Predictions from the model")

    return y_preds
