import uvicorn
from fastapi import FastAPI
# from mangum import Mangum

from model_loader import ModelLoader
from model_parameters import TrainParameters
from predict_request import PredictRequest, ModelResponse

app = FastAPI()
# handler = Mangum(app=app) # used for AWS lambda

classifier = ModelLoader('random forest')


# @app.get("/", response_model=ModelResponse)
# async def root() -> ModelResponse:
#     return ModelResponse(error="This is a test endpoint.")

@app.get("/")
async def root():
    return {"This is a test endpoint."}


@app.get("/predict", response_model=ModelResponse)
async def explain_api() -> ModelResponse:
    return ModelResponse(
        error="Send a POST request to this endpoint with 'features' data."
    )


@app.post("/train")
async def train(params: TrainParameters):
    print("Model Training Started")
    app.model = ModelLoader(params.model.lower())
    return True


@app.post("/predict", response_model=ModelResponse)
async def get_model_prediction(request: PredictRequest):
    print("Predicting")
    prediction = classifier.predict(request.features)

    # return ModelResponse(
    #     prediction=prediction
    # )
    return prediction


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    # uvicorn api:app --reload
