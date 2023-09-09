"""
Render Api module test
"""
import requests
import json


data = {
    " age": 29,
    " hours_per_week": 70,
    " workclass": "Private",
    "education": "Some-college",
    " marital_status": "Married-civ-spouse",
    " occupation": "Exec-managerial",
    " relationship": "Husband",
    " race": "Black",
    " sex": "Male",
    " native_country": "United-States"
}

r = requests.post('https://census-project.onrender.com/', json=data)


print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

# print("Response code: %s" % r.status_code)
# print("Response body: %s" % r.json())