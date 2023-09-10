"""
Render Api module test
"""
import requests

# Define the API endpoint
# endpoint = "https://census-project.onrender.com/"
endpoint = "http://127.0.0.1:8000/"

# Sample data for testing (adjust this as per your needs)
data = {
    "age": 34,
    "workclass": "Private",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

response = requests.post(endpoint, json=data)

# Print the server response
# print(response.json())

print(response.text)
