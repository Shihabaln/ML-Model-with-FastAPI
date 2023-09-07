
"""
Test script for the basic_cleaning.py module
"""
import pytest
from ml.basic_cleaning import load_data

@pytest.fixture
def cleaned_data():
    """Load and clean dataset fixture."""
    # Adjust this path to the path of your data file
    input_path = '../data/prepared_census.csv'
    data = load_data(input_path)
    # Here, you might want to apply any other cleaning functions from basic_cleaning
    return data

def test_null(cleaned_data):
    """Data is assumed to have no null values."""
    assert cleaned_data.shape == cleaned_data.dropna().shape

def test_question_mark(cleaned_data):
    """Data is assumed to have no question marks value."""
    assert '?' not in cleaned_data.values

def test_removed_columns(cleaned_data):
    """Data is assumed to have no certain columns."""
    # Check for columns that you expect to be removed after cleaning
    assert "fnlgt" not in cleaned_data.columns
    assert "education-num" not in cleaned_data.columns
    assert "capital-gain" not in cleaned_data.columns
    assert "capital-loss" not in cleaned_data.columns
