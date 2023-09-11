"""
Unit test of main.py API module with pytest
author: Francisco Nogueira
"""

from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)


def test_root():
    """
    Test welcome message - GET
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'message': 'Census Project  - Welcome to my page!'}

def test_inference_case1():
    """
    Test model inference output for one case - POST
    """
    sample =  {"age": 30,
                "workclass": 'State-gov',
                "education": 'Bachelors',
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "capital_gain": 0,
                "capital_loss": 0,
                "sex": "Male",
                "capital_gain": 5000,
                "hours_per_week": 30,
                "native_country": 'Portugal'}

    data = json.dumps(sample)

    r = client.post("/", data=data )

    # test response and output
    assert r.status_code == 200

    # test prediction vs expected label
    assert r.json()['class_name'] == " <=50K"


def test_inference_case2():
    """
    Test model inference output for one case - POST
    """
    sample =    {'age':55,
                'workclass':"Private", 
                'education':"Doctorate",
                'education_num':16,
                'marital_status':"Separated",
                'occupation':"Exec-managerial",
                'relationship':"Not-in-family",
                'race':"Black",
                "capital_gain": 0,
                "capital_loss": 0,
                'sex':"Male",
                'hours_per_week':50,
                'native_country':"United-States"
            }

    data = json.dumps(sample)

    r = client.post("/", data=data )

    # test response and output
    assert r.status_code == 200

    # test prediction vs expected label
    assert r.json()['class_name'] == ' <=50K'