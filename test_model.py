"""
Unit test of model.py module with pytest
"""

import pytest, os, logging, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix

from ml.model_1 import inference, compute_model_metrics
from ml.data_1 import process_data


"""
Fixture - The test functions will 
use the return of data() as an argument
"""
@pytest.fixture(scope="module")
def data():
    # code to load in the data.
    datapath = "./data/prepared_census.csv"
    return pd.read_csv(datapath)


@pytest.fixture(scope="module")
def path():
    return "./data/prepared_census.csv"


@pytest.fixture(scope="module")
def features():
    """
    Fixture - will return the categorical features as argument
    """
    cat_features = [    "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """
    Fixture - returns cleaned train dataset to be used for model testing
    """
    train, test = train_test_split( data, 
                                test_size=0.20, 
                                random_state=1, 
                                stratify=data['salary']
                                )
    X_train, y_train, encoder, lb = process_data(
                                            train,
                                            categorical_features=features,
                                            label="salary",
                                            training=True
                                        )
    return X_train, y_train


"""
Test methods
"""
def test_import_data(path):
    """
    Test presence and shape of dataset file
    """
    try:
        df = pd.read_csv(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_features(data, features):
    """
    Check that categorical features are in dataset
    """
    try:
        assert sorted(set(data.columns).intersection(features)) == sorted(features)
    except AssertionError as err:
        logging.error(
        "Testing dataset: Features are missing in the data columns")
        raise err


def test_is_model():
    """
    Check saved model is present
    """
    savepath = "./model/model.joblib"
    if os.path.isfile(savepath):
        try:
            _ = joblib.load(open(savepath, 'rb'))
        except Exception as err:
            logging.error(
            "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pass


def test_is_fitted_model(train_dataset):
    """
    Check saved model is fitted
    """

    X_train, y_train = train_dataset
    savepath = "./model/model.joblib"
    model = joblib.load(open(savepath, 'rb'))

    try:
        model.predict(X_train)
    except NotFittedError as err:
        logging.error(
        f"Model is not fit, error {err}")
        raise err


def test_inference(train_dataset):
    """
    Check inference function
    """
    X_train, y_train = train_dataset

    savepath = "./model/model.joblib"
    if os.path.isfile(savepath):
        model = joblib.load(open(savepath, 'rb'))

        try:
            preds = inference(model, X_train)
        except Exception as err:
            logging.error(
            "Inference cannot be performed on saved model and train data")
            raise err
    else:
        pass


def test_compute_model_metrics(train_dataset):
    """
    Check calculation of performance metrics function
    """
    X_train, y_train = train_dataset

    savepath = "./model/model.joblib"
    if os.path.isfile(savepath):
        model = joblib.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:
            precision, recall, fbeta = compute_model_metrics(y_train, preds)
        except Exception as err:
            logging.error(
            "Performance metrics cannot be calculated on train data")
            raise err
    else:
        pass

def test_compute_confusion_matrix(train_dataset):
    """
    Check calculation of confusion matrix function
    """
    X_train, y_train = train_dataset

    savepath = "./model/model.joblib"
    if os.path.isfile(savepath):
        model = joblib.load(open(savepath, 'rb'))
        preds = inference(model, X_train)

        try:

            cm = confusion_matrix(y_train, preds)
        except Exception as err:
            logging.error(
            "Confusion matrix cannot be calculated on train data")
            raise err
    else:
        pass