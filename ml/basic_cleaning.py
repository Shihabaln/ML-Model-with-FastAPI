"""
Basic data cleaning module
"""
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_data(input_path):
    """
    Load data from a CSV file.

    Parameters:
    input_path (str): The path to the input CSV file.

    Returns:
    df (DataFrame): The loaded data frame.
    """
    try:
        df = pd.read_csv(input_path, encoding="utf-8", engine="python")
        return df
    except FileNotFoundError:
        logger.error(f"File {input_path} not found.")
        return None


def save_data(df, output_path):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    df (DataFrame): The data frame to save.
    output_path (str): The path to the output CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
def strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespaces from string columns.

    Parameters:
    - df (pd.DataFrame): Input data.

    Returns:
    - pd.DataFrame: Data with stripped string columns.
    """
    # Identify object columns
    obj_cols = df.columns[df.dtypes == "object"]

    # Strip whitespaces
    for col in obj_cols:
        df[col] = df[col].apply(lambda rows: rows.strip())
    df.columns = df.columns.str.strip()
    return df

def run(
    args=None,
    input_artifact="../data/census.csv",
    output_artifact="../data/prepared_census.csv",
):
    """
    Main function to run the basic data cleaning steps.

    Parameters:
    args (Namespace, optional): Command line arguments (if any).
    input_artifact (str): The path to the input CSV file.
    output_artifact (str): The path to the output CSV file.
    """
    if args is not None:
        _input_artifact = args.input_artifact
        _output_artifact = args.output_artifact
    else:
        _input_artifact = input_artifact
        _output_artifact = output_artifact

    df = load_data(_input_artifact)
    # Cleaning the data
    logger.info("Start to cleaning data...")
    try:
        df.replace({"?": None}, inplace=True)
        df.dropna(inplace=True)
        df.drop("fnlgt", axis="columns", inplace=True)
        df.drop("education-num", axis="columns", inplace=True)
        df.drop("capital-gain", axis="columns", inplace=True)
        df.drop("capital-loss", axis="columns", inplace=True)

    except Exception as e:
        logger.error(e)
    # deleting white spaces   
    df = strip_string_columns(df)
    # Save the results to a CSV file
    logger.info("Saving preprocessing data to CSV")

    save_data(df, output_artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning")
    parser.add_argument("--input_artifact", type=str, help="Input CSV file")
    parser.add_argument("--output_artifact", type=str, help="Output CSV file")
    args = parser.parse_args()
    _ = run(args)
