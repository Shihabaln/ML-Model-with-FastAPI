
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def get_categorical_features():
    """
    Return a list of default categorical features.
    """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

def handle_features(X, categorical_features):
    """
    Handle categorical and continuous features.
    
    Parameters:
    X (DataFrame): The data frame containing features.
    categorical_features (list): The list of categorical feature names.
    
    Returns:
    X_categorical (array): Array of categorical features.
    X_continuous (DataFrame): DataFrame of continuous features.
    categorical_features (list): Updated list of categorical features.
    """
    if len(categorical_features) == 0:
        categorical_features = get_categorical_features()
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features)
    return X_categorical, X_continuous, categorical_features

def handle_label(X, label):
    """
    Handle label data.
    
    Parameters:
    X (DataFrame): The data frame containing features and possibly labels.
    label (str): The name of the label column.
    
    Returns:
    X (DataFrame): The data frame without label column.
    y (array or DataFrame): The label data.
    """
    if label:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    return X, y

def transform_features(X_categorical, X_continuous, training, encoder, lb, y):
    """
    Transform features and labels.
    
    Parameters:
    X_categorical (array): Array of categorical features.
    X_continuous (DataFrame): DataFrame of continuous features.
    training (bool): Flag to indicate if it's training phase.
    encoder (OneHotEncoder): Pre-trained OneHotEncoder (used if training=False).
    lb (LabelBinarizer): Pre-trained LabelBinarizer (used if training=False).
    y (array or DataFrame): The label data.
    
    Returns:
    X (array): Transformed feature data.
    y (array): Transformed label data.
    encoder (OneHotEncoder): Fitted or pre-trained OneHotEncoder.
    lb (LabelBinarizer): Fitted or pre-trained LabelBinarizer.
    """
    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError as e:
            logger.error(e)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb

def process_data(X, categorical_features=[], label=None, training=True, encoder=None, lb=None):
    """
    Process the data used in the machine learning pipeline.
    
    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.
    
    Parameters:
    X (DataFrame): The data frame containing features and possibly labels.
    categorical_features (list): The list of categorical feature names.
    label (str): The name of the label column.
    training (bool): Flag to indicate if it's training phase.
    encoder (OneHotEncoder): Pre-trained OneHotEncoder (used if training=False).
    lb (LabelBinarizer): Pre-trained LabelBinarizer (used if training=False).
    
    Returns:
    X (array): Transformed feature data.
    y (array): Transformed label data.
    encoder (OneHotEncoder): Fitted or pre-trained OneHotEncoder.
    lb (LabelBinarizer): Fitted or pre-trained LabelBinarizer.
    """
    X, y = handle_label(X, label)
    X_categorical, X_continuous, categorical_features = handle_features(X, categorical_features)
    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return transform_features(X, training, encoder, lb, y)
