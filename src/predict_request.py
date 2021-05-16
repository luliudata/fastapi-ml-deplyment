from pydantic import BaseModel
from typing import Optional, List, Dict


class PredictRequest(BaseModel):
    features: Dict[str, float]


class ModelResponse(BaseModel):
    prediction_class: Optional[str]
    probability: Optional[float]
    # prediction_class: Optional[List[Dict[str, float]]]
    # error: Optional[str]
