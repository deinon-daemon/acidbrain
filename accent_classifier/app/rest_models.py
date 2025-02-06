from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    prediction: str | List[str]
    score: float
    filename: str | None = None
    probabilities: list | None = None
    embeddings: list | None = None


class BatchPredictionRequest(BaseModel):
    file_urls: List[str]
