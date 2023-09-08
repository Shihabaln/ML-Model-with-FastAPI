"""
Model steps module
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    make_scorer,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()




def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Use GridSearch for hyperparameter tuning and cross-validation

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameters = {
        'n_estimators': [10, 20, 30],
        'max_depth': [5, 10],
        'min_samples_split': [20, 50, 100],
        'learning_rate': [1.0],  # 0.1,0.5,
    }

    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching best hyperparameters on {} cores".format(njobs))

    clf = GridSearchCV(GradientBoostingClassifier(random_state=0),
                       param_grid=parameters,
                       cv=3,
                       n_jobs=njobs,
                       verbose=2,
                       )

    clf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {}".format(clf.best_params_))

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
