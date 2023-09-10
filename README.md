Udacity Census Project

### Project Overview 
This project focuses on establishing a pipeline for training a machine learning model and 
subsequently publishing it as a public API on Render.

### Setup 
pip install -r requirements.txt

### Machine Learning Pipeline
python ml_pipeline.py

### Usage 
```
usage: ml_pipeline.py [-h] [--steps {basic_cleaning,train_model,evaluate_model,all}]

ML Training Pipeline

optional arguments:
  -h, --help            Display this help message and exit.
  --steps {basic_cleaning,train_model,evaluate_model,all}
                        Specify the pipeline steps to execute.
```

The processed data is stored at the specified path, while the trained model and 
other artifacts are saved to another designated path

### Tests 
 Running test case on script: `pip install -e . && pytest .` to install test module and execute pytest script.

 ### API
 To launch the API, run: 
 `https://census-project.onrender.com`
 Access the provided link [link](https://census-project.onrender.com)to utilize the Swagger UI and test the API.

 **Check Render deployed API**
- Running script: `python render_api_test.py`
- The stdout should be:
```bash
Response 
Response 
```