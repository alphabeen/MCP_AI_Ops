from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    input_sequence: List[float]

class PredictResponse(BaseModel):
    prediction: float
