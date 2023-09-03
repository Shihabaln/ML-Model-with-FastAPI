
"""
Train machine learning model
"""
import pandas as pd
import joblib
import logging
import argparse

from starter.ml.data import process_data
from starter.ml.model import train_model
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def load_and_split_data(data_file):
    """
    Load and split data for training and testing.
    
    Parameters:
    data_file (str): The path to the input CSV file.
    
    Returns:
    X_train (DataFrame): The training feature data.
    X_test (DataFrame): The test feature data.
    y_train (array): The training labels.
    y_test (array): The test labels.
    """
    df = pd.read_csv(data_file)
    X = df.drop(['salary'], axis=1)
    y = df['salary']
    return train_test_split(X, y, test_size=0.20, random_state=42)

def run(
    args=None,
    data_file="./data/prepared_census.csv",
    output_model="./model/model.joblib",
    output_encoder="./model/encoder.joblib",
    output_lb="./model/lb.joblib"
):
    """
    Main function to run the model training steps.
    
    Parameters:
    args (Namespace, optional): Command line arguments (if any).
    data_file (str): The path to the input CSV file.
    output_model (str): The path to the output model file.
    output_encoder (str): The path to the output encoder file.
    output_lb (str): The path to the output label binarizer file.
    """
    if args:
        data_file = args.data_file
        output_model = args.output_model
        output_encoder = args.output_encoder
        output_lb = args.output_lb
    
    X_train, X_test, y_train, y_test = load_and_split_data(data_file)
    
    # Process the data
    X_train, y_train, encoder, lb = process_data(
        X_train, categorical_features=[], label=None, training=True
    )
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save artifacts
    joblib.dump(model, output_model)
    joblib.dump(encoder, output_encoder)
    joblib.dump(lb, output_lb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train machine learning model")
    parser.add_argument('--data_file', type=str, help="Input data file")
    parser.add_argument('--output_model', type=str, help="Output model file")
    parser.add_argument('--output_encoder', type=str, help="Output encoder file")
    parser.add_argument('--output_lb', type=str, help="Output label binarizer file")
    args = parser.parse_args()
    run(args)
