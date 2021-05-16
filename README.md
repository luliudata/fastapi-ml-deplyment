# Machine Learning Model Deployment with FastAPI

This is my first time using FastAPI, it is fun to learn something new.
See documentation: https://fastapi.tiangolo.com/

Git repo: https://github.com/tiangolo/fastapi

This repository contains data analysis for the UCI heart failure clinical records dataset,
model training with a couple of commonly-used classifiers as well as an ensemble voting classifier (only with default parameters).
Then the chosen RandomForest Classifier is deployed with FastAPI to make predictions.

Prerequisites
---------
To setup the virtual environment with Anaconda 
(The commands below have been tested with conda version both 4.7.10 and 4.9.2.)

```
conda create -n py38 python=3.8 anaconda
```
Note that `py38` is self-defined environment name.

To activate the virtual environment
```
conda activate py38
```
Please note that all the required dependencies can be found in the `requirments.txt` under `src` folder

Sample Request
---------

To run the server:

```
uvicorn api:app --reload
```
If you see the last line of the results is:
```
Application startup complete
```
Copy `http://127.0.0.1:8000/` to a web browser (e.g: Google Chrome), you'll see a default `root` page with a pre-defined message:
```
["This is a test endpoint."]
```

Then adds `/docs` at the end of `http://127.0.0.1:8000/`, the Swagger page will show up, which provides an interactive page to make model predictions.

The RandomForest Classifier is trained to classify if the patient died during the follow-up period (0 - Survival / 1 - dead).
Pass a testing JSON payload:

```
{
   "features":{
      "age":20,
      "anaemia":0,
      "creatinine_phosphokinase":800,
      "diabetes":0,
      "ejection_fraction":20,
      "high_blood_pressure":0,
      "platelets":150000,
      "serum_creatinine":1.5,
      "serum_sodium":110,
      "sex":1,
      "smoking":0,
      "time":3
   }
}
```
The response you'll get looks something like this:

```
{
  "prediction_class": "dead",
  "probability": 0.91
}
```

The sample request is:
```
curl -X 'POST' \   'http://127.0.0.1:8000/predict' \   -H 'accept: application/json' \   -H 'Content-Type: application/json' \   -d '{   "features": {     "age": 20,     "anaemia": 0,     "creatinine_phosphokinase": 800,     "diabetes": 0,     "ejection_fraction": 20,     "high_blood_pressure": 0,     "platelets": 150000,     "serum_creatinine": 1.5,     "serum_sodium": 110,     "sex": 1,     "smoking": 0,     "time": 3   } }'
```

Future Work
---------
Due to the time constraints and since this is the very first time I am using FastAPI, I had to spend some time to understand how FastAPI works,
the functionality for this end-to-end deployment is not complete yet, there are a few more directions to explore, here are some thoughts:

1. Package the scripts and host it on AWS Lambda with API Gateway integration, we can have a completely serverless API
infrastructure with many benefits, it is scalable and robust, also it can be cost-effective if it is for smaller APIs 
that might not have over millions of requests.
2. We can also use AWS StepFunction to orchestrate different stages: 

- Data preprocessing(depends on where our data comes from, we can upload them to S3 bucket, use Lambda to pre-process them)
- The clean data will also be stored in S3, then fed into the machine learning model which is hosted on AWS SageMaker, 
we can either use 'batch transform jobs' (for processing large batch data) or 'endpoint'(for processing realtime data) to get the predictions,
(the model itself can either be trained on SageMaker or we can use a pre-trained model.) 

The two stages above can be linked via StepFunction, which also provides a serverless infrastructure, and with error handling, retry logic etc.