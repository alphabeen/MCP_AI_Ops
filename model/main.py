from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from .predict import predict_lstm

app = FastAPI(title="MCP AI 예측 서버")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = predict_lstm(request.input_sequence)
    return PredictResponse(prediction=result)
